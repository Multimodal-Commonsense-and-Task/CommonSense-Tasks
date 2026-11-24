from MTV.mtv_utils import *
from MTV.preprocess import *
from PIL import Image
import torch
import copy
import numpy as np
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
import types
from pathlib import Path

# ==========================================
# Common Image Utilities
# ==========================================

def load_image(image_file):
    try:
        if isinstance(image_file, str):
            image_path = Path(image_file)
            if not image_path.is_file() and len(image_path.parts) >= 3 and image_path.parts[0] in ("data", "."):
                parts = list(image_path.parts)
                if parts[0] == ".":
                    parts = parts[1:]
                if len(parts) >= 3 and parts[0] == "data":
                    nested = Path(parts[0]) / parts[1] / parts[1] / Path(*parts[2:])
                    if nested.is_file():
                        image_path = nested
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_file.convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_file}: {e}")
        return None
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        if image is not None:
            out.append(image)
    
    if not out and image_files: 
        raise ValueError(f"Failed to load any images from: {image_files}")
        
    return out

# --- InternVL Specific Preprocessing ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def find_attn_o_proj_names(model):
    names = []
    for name, _ in model.named_modules():
        if name.endswith("self_attn.o_proj"):
            names.append(name)
    if not names:
        raise RuntimeError("No self_attn.o_proj found")
    return names

# ==========================================
# Base Helper Class
# ==========================================

class ModelHelper:
    def __init__(self):
        pass
    def insert_image(self, text, image_list):
        pass
    def forward(self, model_input, labels=None):
        pass
    def generate(self, model_input, max_new_tokens):
        pass

# ==========================================
# 1. InternVL Helper
# ==========================================

class InternVLHelper(ModelHelper):
    def __init__(self, model, tokenizer, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        
        if not hasattr(model.language_model, "_original_forward"):
            model.language_model._original_forward = model.language_model.forward

        def forward_with_flags(self, *args, image_flags=None, **kwargs):
            return self._original_forward(*args, **kwargs)

        model.language_model.forward = types.MethodType(forward_with_flags, model.language_model)

        if not hasattr(model, 'img_context_token_id') or model.img_context_token_id is None:
            img_token = "<IMG_CONTEXT>"
            if img_token not in tokenizer.get_vocab():
                tokenizer.add_tokens([img_token], special_tokens=True)
                model.resize_token_embeddings(len(tokenizer))
            
            token_id = tokenizer.convert_tokens_to_ids(img_token)
            model.img_context_token_id = token_id

        self.model_config = {
            "n_heads": model.language_model.config.num_attention_heads,
            "n_layers": model.language_model.config.num_hidden_layers,
            "resid_dim": model.language_model.config.hidden_size,
            "name_or_path": model.language_model.config._name_or_path,
            "attn_hook_names": [f'language_model.model.layers.{layer}.attention.wo' for layer in range(model.language_model.config.num_hidden_layers)],
            "layer_hook_names": [f'language_model.model.layers.{layer}' for layer in range(model.language_model.config.num_hidden_layers)]
        }
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3 
        self.nonspecial_idx = 0
        self.transform = build_transform(input_size=448)

    def insert_image(self, text, image_list, gt=None):
        images = load_images(image_list)
        if not images:
            raise ValueError(f"InternVLHelper: No images provided.")

        pixel_values_list = []
        total_visual_tokens = [] 

        for image in images:
            processed_images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
            pixel_values = [self.transform(img) for img in processed_images]
            pixel_values = torch.stack(pixel_values)
            pixel_values_list.append(pixel_values)
            
            num_tiles = pixel_values.size(0)
            total_visual_tokens.append("<IMG_CONTEXT>" * (256 * num_tiles))
        
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).to(self.model.device).to(self.model.dtype)
            image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long).to(self.model.device)
        else:
            raise ValueError("InternVLHelper: Failed to generate pixel_values.")

        parts = text.split("<image>")
        new_text = ""
        for i, part in enumerate(parts):
            new_text += part
            if i < len(total_visual_tokens):
                new_text += total_visual_tokens[i]
        
        tokenized = self.tokenizer(new_text, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(self.model.device)
        attention_mask = tokenized['attention_mask'].to(self.model.device)
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "pixel_values": pixel_values, 
            "image_flags": image_flags
        }

    def forward(self, model_input, labels=None):
        return self.model(
            input_ids=model_input["input_ids"],
            attention_mask=model_input["attention_mask"],
            pixel_values=model_input["pixel_values"],
            image_flags=model_input["image_flags"],
            labels=labels
        )

    def generate(self, model_input, max_new_tokens):
        outputs = self.model.generate(
            input_ids=model_input["input_ids"],
            attention_mask=model_input.get("attention_mask", None),
            pixel_values=model_input["pixel_values"],
            image_flags=model_input["image_flags"],
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        input_len = model_input["input_ids"].shape[1]
        output_ids = outputs[:, input_len:]
        
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

class Internvl3_5Helper(InternVLHelper):
    def __init__(self, model, tokenizer, cur_dataset):
        super().__init__(model, tokenizer, cur_dataset)

        lm = model.language_model.model
        config = lm.config

        self.model_config = {
            "n_heads": config.num_attention_heads,
            "n_layers": config.num_hidden_layers,
            "resid_dim": config.hidden_size,
            "name_or_path": config._name_or_path,
            "attn_hook_names": [
                f"language_model.model.layers.{i}.self_attn.o_proj"
                for i in range(config.num_hidden_layers)
            ],
            "layer_hook_names": [
                f"language_model.model.layers.{i}"
                for i in range(config.num_hidden_layers)
            ],
        }

        self.split_idx = 3

# ==========================================
# 2. Qwen2.5-VL Helper (Separated Processor/Tokenizer)
# ==========================================

class Qwen2_5_Helper(ModelHelper):
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        self.model_config = {
            "n_heads": model.model.config.num_attention_heads,
            "n_layers": model.model.config.num_hidden_layers,
            "resid_dim": model.model.config.hidden_size,
            "name_or_path": model.model.config._name_or_path,
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
            "layer_hook_names": [f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)]
        }
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0

    def insert_image(self, text, image_list, gt=None):
        images = load_images(image_list)
        if not images:
            raise ValueError(f"Qwen2_5_Helper: No images provided.")

        messages = []
        content = []
        parts = text.split("<image>")
        img_idx = 0
        
        for part in parts:
            if part:
                content.append({"type": "text", "text": part})
            if img_idx < len(images):
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
        
        messages.append({"role": "user", "content": content})
        
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=images, 
            videos=None, 
            padding=True,
            return_tensors="pt",
        )
        return inputs.to("cuda")

    def forward(self, model_input, labels=None):
        return self.model(**model_input, labels=labels)

    def generate(self, model_input, max_new_tokens):
        generated_ids = self.model.generate(**model_input, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_input.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

class Qwen3_Helper(ModelHelper):
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer

        attn_hook_names = find_attn_o_proj_names(model)

        text_config = model.config.text_config
        n_heads = text_config.num_attention_heads

        self.model_config = {
            "n_heads": n_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": text_config.hidden_size,
            # Qwen3-VL sets head_dim independently of hidden_size (e.g. the 4B has
            # hidden_size 2560 with 32 heads of 128), so do not assume resid_dim / n_heads.
            "head_dim": getattr(text_config, "head_dim", None) or text_config.hidden_size // n_heads,
            "attn_hook_names": attn_hook_names,
        }

        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 0

    def insert_image(self, text, image_list, gt=None):
        images = load_images(image_list)

        messages = []
        content = []
        parts = text.split("<image>")
        img_idx = 0

        for part in parts:
            if part:
                content.append({"type": "text", "text": part})
            if img_idx < len(images):
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1

        messages.append({"role": "user", "content": content})

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=images,
            padding=True,
            return_tensors="pt"
        )

        return inputs.to(self.model.device)

    def forward(self, model_input, labels=None):
        return self.model(**model_input, labels=labels)

    def generate(self, model_input, max_new_tokens):
        generated_ids = self.model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(model_input.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(
            trimmed, skip_special_tokens=True
        )[0]

# ==========================================
# 3. LLaVA-OneVision Helper (Separated Processor/Tokenizer)
# ==========================================

class OneVisionHelper(ModelHelper):
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        self.model_config = {
            "n_heads": model.language_model.model.config.num_attention_heads,
            "n_layers": model.language_model.model.config.num_hidden_layers,
            "resid_dim": model.language_model.model.config.hidden_size,
            "name_or_path": model.language_model.model.config._name_or_path,
            "attn_hook_names": [f'language_model.model.layers.{layer}.self_attn.o_proj' for layer in range(model.language_model.model.config.num_hidden_layers)],
            "layer_hook_names": [f'language_model.model.layers.{layer}' for layer in range(model.language_model.model.config.num_hidden_layers)]
        }
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 0

    def insert_image(self, text, image_list, gt=None):
        images = load_images(image_list)
        if not images:
            raise ValueError(f"OneVisionHelper: No images provided.")

        if "<image>" in text:
            parts = text.split("<image>")
            content = []
            img_idx = 0
            for part in parts:
                if part:
                    content.append({"type": "text", "text": part})
                if img_idx < len(images):
                    content.append({"type": "image"})
                    img_idx += 1
        else:
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]

        conversation = [{"role": "user", "content": content}]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors='pt').to(self.model.device, torch.float16)
        return inputs

    def forward(self, model_input, labels=None):
        return self.model(**model_input, labels=labels)

    def generate(self, model_input, max_new_tokens):
        out = self.model.generate(**model_input, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

class OneVision1_5Helper(ModelHelper):
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer

        lm = model.language_model
        config = lm.config

        attn_hook_names = [
            name for name, _ in model.named_modules()
            if name.endswith("self_attn.o_proj")
        ]

        if len(attn_hook_names) == 0:
            raise RuntimeError("No self_attn.o_proj found in OneVision1.5 model")

        self.model_config = {
            "n_heads": config.num_attention_heads,
            "n_layers": len(attn_hook_names),
            "resid_dim": config.hidden_size,
            "attn_hook_names": attn_hook_names,
        }

        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 0

    def insert_image(self, text, image_list, gt=None):
        images = load_images(image_list)

        content = []
        parts = text.split("<image>")
        img_idx = 0

        for part in parts:
            if part:
                content.append({"type": "text", "text": part})
            if img_idx < len(images):
                content.append({"type": "image"})
                img_idx += 1

        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        return self.processor(
            images=images,
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

    def forward(self, model_input, labels=None):
        return self.model(**model_input, labels=labels)

    def generate(self, model_input, max_new_tokens):
        out = self.model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

# ==========================================
# 4. Idefics3 Helper
# ==========================================

class Idefics3Helper(ModelHelper):
    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_image_splitting = False 
            print("[Idefics3Helper] Disabled image splitting to save memory.")

        self.model_config = {
            "n_heads": model.config.text_config.num_attention_heads,
            "n_layers": model.config.text_config.num_hidden_layers,
            "resid_dim": model.config.text_config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "attn_hook_names": [f'model.text_model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.text_config.num_hidden_layers)],
            "layer_hook_names": [f'model.text_model.layers.{layer}' for layer in range(model.config.text_config.num_hidden_layers)]
        }
        
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 0

    def insert_image(self, text, image_list, gt=None):
        images = load_images(image_list)
        if not images:
            raise ValueError(f"Idefics3Helper: No images provided.")
        
        content = []
        parts = text.split("<image>")
        img_idx = 0
        for part in parts:
            if part:
                content.append({"type": "text", "text": part})
            if img_idx < len(images):
                content.append({"type": "image"})
                img_idx += 1
                
        prompts = [{"role": "user", "content": content}]
        formatted_text = self.processor.apply_chat_template(prompts, add_generation_prompt=True)
        inputs = self.processor(text=[formatted_text], images=images, return_tensors="pt").to(self.model.device)
        return inputs

    def forward(self, model_input, labels=None):
        return self.model(**model_input, labels=labels)

    def generate(self, model_input, max_new_tokens):
        out = self.model.generate(**model_input, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, model_input['input_ids'].shape[1]:], skip_special_tokens=True)[0]

# ==========================================
# 5. Otter Helper
# ==========================================

class OtterHelper(ModelHelper):
    def __init__(self, model, tokenizer, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer

        lang_model = getattr(model, 'lang_encoder', None)
        if lang_model is None: 
            lang_model = model
        
        config = lang_model.config
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        resid_dim = config.hidden_size

        attn_hook_names = [f'lang_encoder.model.layers.{layer}.decoder_layer.self_attn.o_proj' for layer in range(n_layers)]
        layer_hook_names = [f'lang_encoder.model.layers.{layer}.decoder_layer' for layer in range(n_layers)]
        
        self.split_idx = 3

        self.model_config = {
            "n_heads": n_heads,
            "n_layers": n_layers,
            "resid_dim": resid_dim,
            "name_or_path": config._name_or_path,
            "attn_hook_names": attn_hook_names,
            "layer_hook_names": layer_hook_names
        }
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.nonspecial_idx = 0

    def insert_image(self, text, image_list, gt=None):
        images = load_images(image_list)
        if not images:
            raise ValueError(f"OtterHelper: No images provided.")
        
        vision_x = []
        transform = T.Compose([
            T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        for img in images:
            vision_x.append(transform(img).unsqueeze(0))
            
        if len(vision_x) > 0:
            vision_x = torch.cat(vision_x, dim=0).unsqueeze(0).unsqueeze(0).to(self.model.device) 
        else:
            raise ValueError("OtterHelper: Failed to process images.")

        lang_x = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return {"vision_x": vision_x, "lang_x": lang_x["input_ids"], "attention_mask": lang_x["attention_mask"]}

    def forward(self, model_input, labels=None):
        return self.model(
            vision_x=model_input["vision_x"],
            lang_x=model_input["lang_x"],
            attention_mask=model_input["attention_mask"],
            labels=labels
        )

    def generate(self, model_input, max_new_tokens):
        out = self.model.generate(
            vision_x=model_input["vision_x"],
            lang_x=model_input["lang_x"],
            attention_mask=model_input["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
