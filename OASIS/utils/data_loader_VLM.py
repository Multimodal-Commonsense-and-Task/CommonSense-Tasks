
import torch
import copy
import os
from torch.utils.data import Dataset
import transformers
from configuration.VLM_config_new import DataArguments
from typing import Dict, Sequence
from PIL import Image
from dataclasses import dataclass
from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava import conversation as conversation_lib_llava
from packaging import version
import shutil
from transformers.trainer_pt_utils import LabelSmoother
import json
from qwen_vl_utils import process_vision_info
import re
import itertools
from typing import Dict, Optional, Sequence, List, Tuple
import random


from models.internvl.conversation import get_conv_template
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from models.internvl.train.dataset import preprocess_internvl2_5
from models.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
# from utils.augment import DataAugmentation
# IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
IS_TOKENIZER_GREATER_THAN_0_14 = True
Image.MAX_IMAGE_PIXELS = None
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
SYSTEM_MESSAGE = "You are a helpful assistant."

### Intern
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


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

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

    

def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value']
        gold = source["conversations"][1]['value']

        if 'llava' in self.data_args.model_name_for_dataarg.lower() or 'llama'in self.data_args.model_name_for_dataarg.lower():
            conv = conversation_lib_llava.default_conversation.copy()
        elif 'bunny' in self.data_args.model_name_for_dataarg.lower():
            conv = conversation_lib_bunny.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = qs

        # if self.preprocess:
        #     image = self.images[index]
        if "image" in source.keys():
            image_file = source["image"]
            
            if isinstance(image_file, list):
                image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
            else:
                image = [Image.open(image_file).convert('RGB')]
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = torch.stack([self.data_args.image_processor.preprocess(expand2square(img, tuple(int(x*255) for x in self.data_args.image_processor.image_mean)), return_tensors='pt')['pixel_values'][0] for img in image])
            else:
                image = torch.stack([self.data_args.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image])
        else: 
            image = torch.zeros(0)
            image_file = []
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return {
            'input_ids':input_ids,
            'image':image,
            'gold':gold,
            'prompt':prompt,
            'image_file':image_file
        }

    def __len__(self):
        return len(self.datalist)
    
class Qwen_GenerationDataset(Dataset):
    def __init__(self, datalist, processor,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(Qwen_GenerationDataset, self).__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None
        
        self.conv = conversation_lib_llava.default_conversation.copy()
        
    def __getitem__(self, index):
        
        sources = self.datalist[index]
        gold = sources["conversations"][1]['value']
        qs = sources["conversations"][0]['value']
        prompt = qs
        is_video = False
        is_image = False
        processor = self.processor
        image_files = []
        if "image" in sources:
            is_image = True
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            messages = copy.deepcopy(llava_to_openai_generate(sources['conversations'],image_files, is_video=is_video))
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None 
            messages = copy.deepcopy(llava_to_openai_generate(sources['conversations'],None, is_video=is_video))
            
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if is_image:
            images, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, padding=False, return_tensors='pt')
            prompt_input_ids = inputs['input_ids']
            pixel_values = inputs[pixel_key]
            image_thw = inputs[grid_key]
        else:
            prompt_input_ids = processor.tokenizer(text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        attention_mask = (prompt_input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=prompt_input_ids,
            prompt=prompt,
        )

        if pixel_key and grid_key:
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict['image_file'] = image_files
            data_dict['gold'] = str(gold)
        else:
            data_dict['image_file'] = image_files
            data_dict['gold'] = str(gold)

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict
        
    def __len__(self):
        return len(self.datalist)


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

@dataclass
class Qwen_DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        if 'pixel_values' in instances[0].keys():
            input_ids, gold, prompt, image_grid_thw = tuple([instance[key] for instance in instances]
                                    for key in ("input_ids", 'gold', 'prompt', 'image_grid_thw'))
        else:
            input_ids, gold, prompt= tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt'))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # Find the length of the longest sequence
        max_length = max(seq.size(0) for seq in input_ids)
        input_ids = torch.stack([
            torch.cat([torch.full((max_length - seq.size(0),), self.tokenizer.pad_token_id), seq])
            for seq in input_ids
        ])
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
            gold=gold,
        )
        if 'pixel_values' in instances[0].keys():
            image_file = [instance['image_file'] for instance in instances]
            batch['image_file'] = image_file
            images = [instance['pixel_values'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                batch['pixel_values'] = images#.reshape(b,n,c,h,w)
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                batch['pixel_values'] = [x.to(dtype=torch.bfloat16) for x in images]
            batch["image_grid_thw"] = image_grid_thw
        batch['prompt'] = prompt
        return batch
    
@dataclass
class DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt, image_file = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt', 'image_file'))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        max_length = max(seq.size(0) for seq in input_ids)
        input_ids = torch.stack([
            torch.cat([torch.full((max_length - seq.size(0),), self.tokenizer.pad_token_id), seq])
            for seq in input_ids
        ])
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
            gold=gold,
            prompt=prompt,
            image_file=image_file
        )
        images = [instance['image'] for instance in instances]
        # if all(x is not None and x.shape == images[0].shape for x in images):
        if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
            images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
            # b, n, c, h, w = images.shape
            # images = images.reshape(b*n,c,h,w)
            # images = self.transform(images).to(dtype=torch.bfloat16)
            batch['images'] = images#.reshape(b,n,c,h,w)
        else:
            # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
            batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

        return batch
# class DataCollatorForGenerationDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances):
#         input_ids, gold, prompt, image_file = tuple([instance[key] for instance in instances]
#                                   for key in ("input_ids", 'gold', 'prompt', 'image_file'))
#         # input_ids = torch.nn.utils.rnn.pad_sequence(
#         #     input_ids,
#         #     batch_first=True,
#         #     padding_value=self.tokenizer.pad_token_id)
#         # input_ids = input_ids[:, :self.tokenizer.model_max_length]

#         max_length = max(seq.size(0) for seq in input_ids)
#         input_ids = torch.stack([
#             torch.cat([torch.full((max_length - seq.size(0),), self.tokenizer.pad_token_id), seq])
#             for seq in input_ids
#         ])
#         input_ids = input_ids[:, :self.tokenizer.model_max_length]
#         attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
#         if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
#             for input_id in input_ids:
#                 input_id[input_id == -300] = self.tokenizer.eos_token_id
#         batch = dict(
#             input_ids=input_ids,#.to(self.device),
#             attention_mask=attention_mask,#.to(self.device),
#             gold=gold,
#             prompt=prompt,
#             image_file=image_file
#         )
#         images = [instance['image'] for instance in instances]
#         # if all(x is not None and x.shape == images[0].shape for x in images):
#         if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
#             images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
#             # b, n, c, h, w = images.shape
#             # images = images.reshape(b*n,c,h,w)
#             # images = self.transform(images).to(dtype=torch.bfloat16)
#             batch['images'] = images#.reshape(b,n,c,h,w)
#         else:
#             # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
#             batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]
#         return batch
# class DataCollatorForGenerationDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances):
#         input_ids, gold, prompt, image_file = tuple([instance[key] for instance in instances]
#                                   for key in ("input_ids", 'gold', 'prompt', 'image_file'))
#         # input_ids = torch.nn.utils.rnn.pad_sequence(
#         #     input_ids,
#         #     batch_first=True,
#         #     padding_value=self.tokenizer.pad_token_id)
#         # input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
#         max_length = max(seq.size(0) for seq in input_ids)
#         input_ids = torch.stack([
#             torch.cat([torch.full((max_length - seq.size(0),), self.tokenizer.pad_token_id), seq])
#             for seq in input_ids
#         ])
#         input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
#         attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
#         if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
#             for input_id in input_ids:
#                 input_id[input_id == -300] = self.tokenizer.eos_token_id
        
#         batch = dict(
#             input_ids=input_ids,#.to(self.device),
#             attention_mask=attention_mask,#.to(self.device),
#             gold=gold,
#             prompt=prompt,
#             image_file=image_file
#         )
#         images = [instance['image'] for instance in instances]
#         # if all(x is not None and x.shape == images[0].shape for x in images):
#         if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
#             images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
#             # b, n, c, h, w = images.shape
#             # images = images.reshape(b*n,c,h,w)
#             # images = self.transform(images).to(dtype=torch.bfloat16)
#             batch['images'] = images#.reshape(b,n,c,h,w)
#         else:
#             # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
#             batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

#         return batch

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future


    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def replace_image_tokens(input_string):

    pattern = r'\n?' + re.escape("<image>") + r'\n?'
    replacement = "<|vision_start|>"+ DEFAULT_IMAGE_TOKEN + "<|vision_end|>"

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False, no_visual=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        if not no_visual:
            transformed_content = replace_image_tokens(conversation["value"])
        else:
            transformed_content = conversation["value"]
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def llava_to_openai_generate(conversations, image_files, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    prompts = []
    for conv_i, conversation in enumerate(conversations):
        if conversation["from"] == "human":
            content = []
            qs = conversation['value']
            qs = qs.replace("\n", " ").strip()

            if "<image> <image>" in qs:
                double_image = True
                chunks = [c.strip() for c in qs.split("<image> <image>") if c.strip()]
                if not qs.startswith("<image> <image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]
            else:
                double_image = False
                chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
                if not qs.startswith("<image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]

            pairs = []

            for chunk in chunks:
                chunk = chunk.strip()

                # optional leading "Question:"
                chunk_wo_question = re.sub(
                    r"^\s*Question\s*[:：]\s*",
                    "",
                    chunk,
                    flags=re.IGNORECASE
                )

                # Find the LAST "Answer:" instead of the first one
                matches = list(re.finditer(r"Answer\s*[:：]\s*", chunk_wo_question, flags=re.IGNORECASE))

                if matches:
                    last_match = matches[-1]
                    q = chunk_wo_question[:last_match.start()].strip()
                    a = chunk_wo_question[last_match.end():].strip()
                else:
                    q, a = chunk_wo_question.strip(), ""

                pairs.append((q.strip(), a.strip()))

            # If the last answer is empty, try to get it from the next gpt turn
            if len(pairs) > 0 and pairs[-1][1] == "":
                if conv_i + 1 < len(conversations) and conversations[conv_i + 1]["from"] == "gpt":
                    pairs[-1] = (pairs[-1][0], conversations[conv_i + 1]["value"].strip())

            if double_image:
                num_pairs = min(len(pairs), len(image_files) // 2)
                for pair_i in range(num_pairs):
                    pair = pairs[pair_i]
                    image1, image2 = image_files[pair_i * 2], image_files[pair_i * 2 + 1]

                    if pair_i == num_pairs - 1:
                        user_dict = {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image1},
                                {"type": "image", "image": image2},
                                {"type": "text", "text": f"User: {pair[0]} \nAssistant: "}
                            ]
                        }
                    else:
                        user_dict = {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image1},
                                {"type": "image", "image": image2},
                                {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"}
                            ]
                        }
                    prompts.append(user_dict)
            else:
                num_pairs = min(len(pairs), len(image_files))
                for image_file_i in range(num_pairs):
                    image_path = image_files[image_file_i]
                    pair = pairs[image_file_i]

                    if image_file_i == num_pairs - 1:
                        user_dict = {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": f"User: {pair[0]} \nAssistant: "}
                            ]
                        }
                    else:
                        user_dict = {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"}
                            ]
                        }
                    prompts.append(user_dict)

    return prompts

class LLM_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path,
            tokenizer,
            data_args,
            processor):
        super(LLM_LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path
        
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None
        self.tokenizer = processor.tokenizer
        
        self.get_rope_index = get_rope_index_25
        
        self.conv = conversation_lib_llava.default_conversation.copy()
        
        assistant_start = self.conv.roles[1]
        eot_token = self.conv.sep
        
        if self.conv.version == 'qwen':
            self.label_start_id = self.tokenizer(assistant_start, return_tensors="pt")['input_ids'][0].tolist()
            self.label_end_id = self.tokenizer(eot_token, return_tensors="pt")['input_ids'][0].tolist()
            
            self.processor.chat_template = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""
        elif self.conv.version == 'llama3':
            self.label_start_id = self.tokenizer(assistant_start, return_tensors="pt")['input_ids'][0][1:].tolist()
            self.label_end_id = self.tokenizer(eot_token, return_tensors="pt")['input_ids'][0][1:].tolist()
        
            self.processor.chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{%- if system_message %}\n    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n    {%- if tools is not none %}\n        {{- "Environment: ipython\\n" }}\n    {%- endif %}\n    {{- "Cutting Knowledge Date: December 2023\\n" }}\n    {{- "Today Date: " + date_string + "\\n\\n" }}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n        {{- "Do not use variables.\\n\\n" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- "\\n\\n" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- "<|eot_id|>" }}\n{%- endif %}\n\n{%- if tools_in_user_message and not tools is none %}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}\n    {%- endif %}\n    {{- "<|start_header_id|>user<|end_header_id|>\\n\\n" }}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>" }}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\\n\\n" + message['content'] | trim + "<|eot_id|>" }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n        {{- '{"name": "' + tool_call.name + '", ' }}\n        {{- '"parameters": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n    {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n{%- endif %}\n"""

    def __len__(self):
        return len(self.list_data_dict)

    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if self.data_args.get_prompt:
            qs = sources["conversations"][0]['value']
        
        # conversation = []
        # for j in range(len(sources["conversations"])):
        #     if j % 2 == 0:
        #         conversation.append({
        #             "role": "user",
        #             "content": str(sources["conversations"][j]['value'])
        #         })
        #     else:
        #         conversation.append({
        #             "role": "assistant",
        #             "content": str(sources["conversations"][j]['value'])
        #         })
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False, no_visual=True))
        prompt = self.processor.apply_chat_template(sources, tokenize=False)
        inputs = self.processor(text=prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        label_ids = input_ids.clone().fill_(IGNORE_INDEX)
        # Find all occurrences of the assistant start and end tokens
        
        assistant_start_tokens = self.label_start_id
        assistant_end_tokens = self.label_end_id
        start_indices = []
        end_indices = []
        for idx in range(len(input_ids[0]) - len(assistant_start_tokens) + 1):
            if input_ids[0][idx:idx + len(assistant_start_tokens)].tolist() == assistant_start_tokens:
                start_indices.append(idx + len(assistant_start_tokens))  # Start of assistant response

        for idx in range(len(input_ids[0]) - len(assistant_end_tokens) + 1):
            if input_ids[0][idx:idx + len(assistant_end_tokens)].tolist() == assistant_end_tokens:
                end_indices.append(idx + len(assistant_end_tokens))  # End of assistant response

        # Match start and end indices to create valid ranges
        for start_idx in start_indices:
            end_idx = next((end for end in end_indices if end > start_idx), None)
            if end_idx is not None:
                label_ids[0][start_idx:end_idx] = input_ids[0][start_idx:end_idx]
        
        data_dict = {
            'input_ids': input_ids,
            'labels': label_ids
        }
        
        if self.data_args.get_prompt:
            data_dict['prompt'] = qs
        
        return data_dict


@dataclass
class LLM_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids[0],#.to(self.device),
            labels=labels,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
            use_cache=False,
        )

        if 'prompt' in instances[0]:
            batch['prompt'] = [instance['prompt'] for instance in instances]

        return batch

@dataclass
class LLM_DataCollatorForGenerationDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, gold, prompt = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", 'gold', 'prompt'))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # Find the length of the longest sequence
        max_length = max(seq.size(0) for seq in input_ids)
        input_ids = torch.stack([
            torch.cat([torch.full((max_length - seq.size(0),), self.tokenizer.pad_token_id), seq])
            for seq in input_ids
        ])
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
            gold=gold,
            prompt=prompt
        )
        if 'pixel_values' in instances[0].keys():
            image_file = [instance['image_file'] for instance in instances]
            batch['image_file'] = image_file
            images = [instance['pixel_values'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            if all(x is not None and x.shape[0] != 0 and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                batch['pixel_values'] = images#.reshape(b,n,c,h,w)
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                batch['pixel_values'] = [x.to(dtype=torch.bfloat16) for x in images]

        return batch
  
class LLM_GenerationDataset(Dataset):
    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 processor=None):
        super(LLM_GenerationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.processor = processor
        self.conv = conversation_lib_llava.default_conversation.copy()
        
        if self.conv.version == 'qwen':
            self.processor.chat_template = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""
        elif self.conv.version == 'llama3':
            self.processor.chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{%- if system_message %}\n    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n    {%- if tools is not none %}\n        {{- "Environment: ipython\\n" }}\n    {%- endif %}\n    {{- "Cutting Knowledge Date: December 2023\\n" }}\n    {{- "Today Date: " + date_string + "\\n\\n" }}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n        {{- "Do not use variables.\\n\\n" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- "\\n\\n" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- "<|eot_id|>" }}\n{%- endif %}\n\n{%- if tools_in_user_message and not tools is none %}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}\n    {%- endif %}\n    {{- "<|start_header_id|>user<|end_header_id|>\\n\\n" }}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>" }}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\\n\\n" + message['content'] | trim + "<|eot_id|>" }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n        {{- '{"name": "' + tool_call.name + '", ' }}\n        {{- '"parameters": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n    {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n{%- endif %}\n"""

    def __getitem__(self, index):
        source = self.datalist[index]
        qs = source["conversations"][0]['value']
        gold = source["conversations"][1]['value']

        conversation = [
            {
            "role": "user",
            "content": qs
            },
        ]
        
        # conversation = [
        #     {
        #     "role": "user",
        #     "content": [{"type":"text","text":qs}
        #                 ]
        #     },
        # ]

        messages = copy.deepcopy(llava_to_openai_generate(source['conversations'],None, is_video=False))
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. " + prompt
        
        if 'image' in source.keys():
            image_file = source['image']
            if isinstance(image_file, list):
                image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
            else:
                image = [Image.open(image_file).convert('RGB')]
        
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                images = [expand2square(img, tuple(int(x*255) for x in self.processor.image_processor.image_mean)) for img in image]
            inputs = self.processor(images=images, text=prompt, return_tensors='pt')
        else:
            inputs = self.processor(text=prompt, return_tensors='pt')
        # return input_ids, pixel_values, gold, prompt, image_file
        return_dict = {
            'input_ids':inputs['input_ids'][0],
            'gold':str(gold),
            'prompt':prompt,
        }
        
        if 'pixel_values' in inputs.keys():
            return_dict['pixel_values'] = inputs['pixel_values']
            return_dict['image_file'] = image_file
        return return_dict

    def __len__(self):
        return len(self.datalist)
    
class Qwen_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(Qwen_LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path
        
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = (256 * 28 * 28)
        self.image_max_pixel = (1280 * 28 * 28)
        self.image_resized_w = None
        self.image_resized_h = None
        self.tokenizer = processor.tokenizer
        
        self.get_rope_index = get_rope_index_25

    def __len__(self):
        return len(self.list_data_dict)
    
    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.processor.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        is_image = False
        if "image" in sources:
            is_image = True
        is_video = False
        
        if self.data_args.get_prompt:
            prompt = sources["conversations"][0]['value']
        
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        
        user_input = sources["conversations"][0]['value']
        gpt_response = sources["conversations"][1]['value']

        processor = self.processor
        if is_image:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
            
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
            
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, no_visual=not(is_image)))
        
        if is_image:
            inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
            prompt_input_ids = inputs['input_ids']
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])
        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
        response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0).to(torch.long)
        labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
        ).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)
        
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        if self.data_args.get_prompt:
            data_dict["prompt"] = prompt_input_ids
            
        return data_dict

def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor



class ImagePreprocess:
    def __init__(self, image_processor, data_args={}):
        self.image_aspect_ratio = getattr(data_args, 'image_aspect_ratio', None)
        self.image_processor = image_processor
        self.image_grid_pinpoints = getattr(data_args, 'image_grid_pinpoints', None)
    
    def __call__(self, image):
        if self.image_aspect_ratio == 'pad':
            image = self.expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        elif self.image_aspect_ratio == "anyres":
            image = self.process_anyres_image(image, self.image_processor, self.image_grid_pinpoints)
            return image
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        return image

    @classmethod
    def expand2square(cls, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    @classmethod
    def process_anyres_image(cls, image, processor, grid_pinpoints):
        """
        Process an image with variable resolutions.

        Args:
            image (PIL.Image.Image): The input image to be processed.
            processor: The image processor object.
            grid_pinpoints (str): A string representation of a list of possible resolutions.

        Returns:
            torch.Tensor: A tensor containing the processed image patches.
        """
        if type(grid_pinpoints) is list:
            possible_resolutions = grid_pinpoints
        else:
            possible_resolutions = ast.literal_eval(grid_pinpoints)
        best_resolution = select_best_resolution(image.size, possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)

        patches = divide_to_patches(image_padded, processor.crop_size['height'])

        image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

        image_patches = [image_original_resize] + patches
        image_patches = [processor(image_patch, return_tensors='pt')['pixel_values'][0]
                        for image_patch in image_patches]
        return torch.stack(image_patches, dim=0)
    
'''
@dataclass
class Intern_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, features, max_item_length=None, pad_id=0):

        first = features[0]
        batch = {}

        batch_lens = [feat['input_ids'].shape for feat in features]
        max_item_length = max_item_length or max(batch_lens)[0]
        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[:feat['labels'].shape[0]] = feat['labels']
            feat['labels'] = temp_labels
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)

            if 'position_ids' in feat:
                temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
                temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
                feat['position_ids'] = temp_position_ids

            if 'loss_weight' in feat:
                temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
                temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
                feat['loss_weight'] = temp_loss_weight

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack([f['label_ids'] for f in features])
            else:
                dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            if k in ('pixel_values', 'image_flags'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.concat([f[k] for f in features])
        return batch
'''


class Intern_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, datalist,
            tokenizer,
            data_args,
            preprocess=False,
            config=None):
        super(Intern_LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.dynamic_image_size = False
        self.is_train = True
        self.group_by_length=False
        self.use_packed_ds=True

        self.config = config
        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))

    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def get_preprocess_function(self):
        preprocess_function = preprocess_internvl2_5
        return preprocess_function

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def load_image(self, image_file):
        image = Image.open(image_file).convert('RGB')
        return image


    def multi_modal_get_item(self, data_item, input_size=448, max_num=12):

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        image_path = data_item['image'][0]
        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)
        transform = self.build_transform(input_size=input_size)
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function("internvl2_5", [copy.deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item, input_size=448, max_num=4):
        transform = self.build_transform(input_size=input_size)
        
        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
                images+=image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function("internvl2_5", [copy.deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def __getitem__(self, i):
        sources = self.datalist[i]
        conv = get_conv_template("internvl2_5")

        if 'image' in sources:
            all_images = sources["image"]
            if not isinstance(all_images, list):
                all_images = [all_images]
                sources["image"] = all_images
            if len(all_images) > 1:
                ret = self.multi_modal_multi_image_get_item(sources)
            else:
                ret = self.multi_modal_get_item(sources)

        return ret

@dataclass
class Intern_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, features, max_item_length=None, pad_id=0):

        first = features[0]
        batch = {}

        batch_lens = [feat['input_ids'].shape for feat in features]
        max_item_length = max_item_length or max(batch_lens)[0]
        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[:feat['labels'].shape[0]] = feat['labels']
            feat['labels'] = temp_labels
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)

            if 'position_ids' in feat:
                temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
                temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
                feat['position_ids'] = temp_position_ids

            if 'loss_weight' in feat:
                temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
                temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
                feat['loss_weight'] = temp_loss_weight

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack([f['label_ids'] for f in features])
            else:
                dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            if k in ('pixel_values', 'image_flags'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.concat([f[k] for f in features])
        return batch

@dataclass
class Qwen_DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            labels=labels,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                batch['images'] = images#.reshape(b,n,c,h,w)
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

        if 'prompt' in instances[0]:
            batch['prompt'] = [instance['prompt'] for instance in instances]

        return batch

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False,
                 perturb_num=0):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.perturb_num = perturb_num
        self.preprocess = preprocess
        self.drop_rate = 0.2
        # self.preprocess = ImagePreprocess(data_args.image_processor, data_args)
        # if preprocess:
        #     self.images = []
        #     for data in self.datalist:
        #         image_file = data['image']
        #         image = Image.open(image_file).convert('RGB')
        #         self.images.append(image)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.datalist:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        sources = self.datalist[i]
        
        # if self.data_args.get_prompt:
        org_source = copy.deepcopy(sources)
        prompt = org_source["conversations"][0]['value']
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            processor = ImagePreprocess(self.data_args.image_processor, self.data_args)
            if self.preprocess:
                image = self.images[i]
            else:
                image_file = self.datalist[i]['image']
                # if ' |sep| ' in image_file:
                if isinstance(image_file, list):
                    image = [Image.open(image_path).convert('RGB') for image_path in image_file] #.split(' |sep| ')
                else:
                    image = [Image.open(image_file).convert('RGB')]

            image = torch.stack([processor(img)for img in image])
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            if self.perturb_num > 0:
                perturb_inputs, perturb_labels = [], []
                for per_i in range(self.perturb_num):
                    text_split = prompt.split()
                    new_text_split = [t for t in text_split if random.uniform(0, 1) > self.drop_rate or "<image>" in t]
                    new_text = " ".join(new_text_split)
                    org_source["conversations"][0]["value"] = new_text
                    perturb_source = preprocess_multimodal(
                    copy.deepcopy([org_source["conversations"]]),
                    self.data_args)
                    data_dict = preprocess_text_llava(
                        perturb_source,
                        self.tokenizer,
                        has_image=('image' in self.datalist[i]))
                    perturb_inputs.append(data_dict["input_ids"][0])
                    perturb_labels.append(data_dict["labels"][0])
                
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            image=torch.zeros(0)
        # if 'llava' in self.data_args.model_name_for_dataarg.lower() or 'qwen' in self.data_args.model_name_for_dataarg.lower() or 'llama' in self.data_args.model_name_for_dataarg.lower():
        data_dict = preprocess_text_llava(
            sources,
            self.tokenizer,
            has_image=('image' in self.datalist[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        # if 'image' in self.datalist[i]:
        data_dict['image'] = image
        
        if self.data_args.get_prompt:
            data_dict['prompt'] = prompt
            
        if self.perturb_num > 0:
            data_dict["perturb"] = perturb_inputs
            data_dict["perturb_labels"] = perturb_labels
        return data_dict
'''
class Intern_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, datalist,
                 tokenizer,
                 data_args,
                 preprocess=False):
        super(Intern_LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        
        
        # self.preprocess = ImagePreprocess(data_args.image_processor, data_args)
        # if preprocess:
        #     self.images = []
        #     for data in self.datalist:
        #         image_file = data['image']
        #         image = Image.open(image_file).convert('RGB')
        #         self.images.append(image)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.datalist:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        conv = get_conv_template("internvl2_5")
        roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
        
        sources = self.datalist[i]     
        
        conversations = []
        conv.messages = []
        source = sources["conversations"]
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
        
            
        if 'image' in sources:
            pixel_values_list = []
            new_conversations = []
            current_image_idx = 0
            all_images = sources["image"]
            if not isinstance(all_images, list):
                all_images = [all_images]
            num_image = len(all_images)
            for conversation in source:
                if conversation['from'] == 'human':
                    image_cnt = conversation['value'].count('<image>')
                    for i in range(image_cnt):
                        image = all_images[i]
                        pixel_values = load_image(image, max_num=12).to(torch.bfloat16)
                        pixel_values_list.append(pixel_values)
                        if current_image_idx == num_image:
                            break
                        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * len(pixel_values)}{IMG_END_TOKEN}'
                        conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                        current_image_idx += 1
                    pixel_values = torch.cat(pixel_values_list, dim=0)  
                new_conversations.append(conversation)
            conversations = new_conversations
            assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'
        
        
        batches, roles = [], []

        for conversation in conversations:
            if conversation['from'] == 'human':
                batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
                roles.append('human')
            elif conversation['from'] == 'gpt':
                batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
                roles.append('gpt')
            else:
                raise NotImplementedError

        add_bos_token = getattr(self.tokenizer, 'add_bos_token', False)
        if add_bos_token:  # for InternLM series
            batches[0] = self.tokenizer.bos_token + batches[0]

        # Tokenize conversations
        input_ids = self.tokenizer(
            batches,
            return_tensors='np',
            padding=False,
            max_length=self.tokenizer.model_max_length,
            truncation=False,
        ).input_ids

        if add_bos_token:  # for InternLM series
            input_ids = [item[1:] for item in input_ids]

        final_input_ids, final_targets = [], []
        ignore_ids = self.tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
        ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
        for role, input_id in zip(roles, input_ids):
            final_input_ids.append(input_id)
            if role == 'system' or role == 'human':
                final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
            elif role == 'gpt':
                target = input_id.copy()
                target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
                target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
                final_targets.append(target)
            else:
                raise NotImplementedError
        input_ids = torch.tensor(np.concatenate(final_input_ids))[:self.tokenizer.model_max_length]
        targets = torch.tensor(np.concatenate(final_targets))[:self.tokenizer.model_max_length]

        # padding = False if group_by_length or use_packed_ds else True
        # if padding:
        current_length = input_ids.size(0)
        padding_length = self.tokenizer.model_max_length - current_length
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=self.tokenizer.pad_token_id)
        targets = torch.nn.functional.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

        input_ids = input_ids.unsqueeze(0)
        targets = targets.unsqueeze(0)

        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=targets,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

'''
'''
class Intern_LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, datalist,
            tokenizer,
            data_args,
            preprocess=False,
            config=None):
        super(Intern_LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.datalist = datalist
        self.data_args = data_args
        self.preprocess = preprocess
        self.dynamic_image_size = False
        self.is_train = True
        self.group_by_length=False
        self.use_packed_ds=False
        
        self.config = config
        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))

    @property
    def lengths(self):
        length_list = []
        for sample in self.datalist:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list
    
    def __len__(self):
        return len(self.datalist)

    def get_preprocess_function(self):
        preprocess_function = preprocess_internvl2_5
        return preprocess_function

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def load_image(self, image_file):
        image = Image.open(image_file).convert('RGB')
        return image 

    
    def multi_modal_get_item(self, data_item, input_size=448, max_num=12):
        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        image_path = data_item['image'][0]
        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            image = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        else:
            image = [image]
            # num_tiles.append(len(image))
        # else:  # Otherwise, use the original image as a single patch
            # num_tiles.append(1)
        transform = self.build_transform(input_size=input_size)
        # images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img).to(dtype=torch.bfloat16) for img in image]
        pixel_values = torch.stack(pixel_values)
        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function("internvl2_5", [copy.deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret
    
    def multi_modal_multi_image_get_item(self, data_item, input_size=448, max_num=4):
        transform = self.build_transform(input_size=input_size)
        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
                images+=image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function("internvl2_5", [copy.deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret
        
    # def multi_modal_multi_image_get_item(self, data_item, input_size=448, max_num=12):

    #     images, num_tiles = [], []
    #     num_image = len(data_item['image'])
        
    #     for image_path in data_item['image']:
    #         # Load the image using tcs_loader if available, otherwise use PIL
    #         image = self.load_image(image_path)
    #         transform = self.build_transform(input_size=input_size)
    #         images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    #         pixel_values = [transform(image) for image in images]
    #         pixel_values = torch.stack(pixel_values)
    #         if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
    #             images+=pixel_values
    #             num_tiles.append(len(pixel_values))
    #         else:  # Otherwise, use the original image as a single patch
    #             images.append(pixel_values)
    #             num_tiles.append(1)
    #     pixel_values = torch.cat(images)
    #     num_patches = pixel_values.size(0)

    #     # Select the appropriate preprocessing function based on the template name
    #     preprocess_function = self.get_preprocess_function()

    #     # Preprocess the conversations and generate the return dictionary
    #     num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
    #     ret = preprocess_function("internvl2_5", [copy.deepcopy(data_item['conversations'])],
    #                               self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
    #                               use_packed_ds=self.use_packed_ds, num_image=num_image)

    #     # Calculate position_ids for packed dataset
    #     position_ids = ret['attention_mask'].long().cumsum(-1) - 1
    #     position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
    #     image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

    #     # Create the final return dictionary
    #     ret = dict(
    #         input_ids=ret['input_ids'][0],
    #         labels=ret['labels'][0],
    #         attention_mask=ret['attention_mask'][0],
    #         position_ids=position_ids[0],
    #         pixel_values=pixel_values,
    #         image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
    #     )
    #     return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.datalist[i]
        conv = get_conv_template("internvl2_5")

        if 'image' in sources:
            all_images = sources["image"]
            if not isinstance(all_images, list):
                all_images = [all_images]
                sources["image"] = all_images
            if len(all_images) > 1:
                ret = self.multi_modal_multi_image_get_item(sources)
            else:
                ret = self.multi_modal_get_item(sources)

        return ret
    '''
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
     
        
        batch = dict(
            input_ids=input_ids,#.to(self.device),
            labels=labels,#.to(self.device),
            attention_mask=attention_mask,#.to(self.device),
        )
        if "perturb" in list(instances[0].keys()):
            perturb_instances = [instance["perturb"] for instance in instances]
            perturb_labels = [instance["perturb_labels"] for instance in instances]
            batch_perturb, batch_perturb_attentions, batch_perturb_labels = [], [], []
            new_perturb_instances, new_perturb_attentions, new_perturb_labels = [], [], []
            for input_ids, labels in zip(perturb_instances, perturb_labels):
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id)
                labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                        batch_first=True,
                                                        padding_value=IGNORE_INDEX)
                input_ids = input_ids[:, :self.tokenizer.model_max_length]
                labels = labels[:, :self.tokenizer.model_max_length]
                
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
                new_perturb_attentions.append(attention_mask)
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    for input_id in input_ids:
                        input_id[input_id == -300] = self.tokenizer.eos_token_id
                new_perturb_instances.append(input_ids)
                new_perturb_labels.append(labels)

            batch_perturb.append(new_perturb_instances)
            batch_perturb_attentions.append(new_perturb_attentions)
            batch_perturb_labels.append(new_perturb_labels)
            batch["perturb"] = batch_perturb
            batch["perturb_attn"] = batch_perturb_attentions
            batch["perturb_labels"] = batch_perturb_labels
            

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                images = torch.stack(images).to(torch.bfloat16)#.to(device=self.device)
                # b, n, c, h, w = images.shape
                # images = images.reshape(b*n,c,h,w)
                # images = self.transform(images).to(dtype=torch.bfloat16)
                batch['images'] = images#.reshape(b,n,c,h,w)
            else:
                # batch['images'] = [self.transform(x.to(device=self.device)).to(dtype=torch.bfloat16) if x.shape[0] != 0 else x.to(torch.bfloat16) for x in images]
                batch['images'] = [x.to(dtype=torch.bfloat16) for x in images]

        if 'prompt' in instances[0]:
            batch['prompt'] = [instance['prompt'] for instance in instances]
            

        return batch

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # if DEFAULT_IMAGE_TOKEN in sentence['value']:
            #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            #     sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
            #     sentence['value'] = sentence['value'].strip()
            if "mmtag" in conversation_lib_llava.default_conversation.version:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


# bunny preprocess

def preprocess_text_bunny(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    if conversation_lib_bunny.default_conversation.sep_style == conversation_lib_bunny.SeparatorStyle.PLAIN:
        return preprocess_plain_bunny(sources, tokenizer)

    if conversation_lib_bunny.default_conversation.version == "bunny":
        return preprocess_bunny(sources, tokenizer, has_image=has_image)
    elif conversation_lib_bunny.default_conversation.version in {"minicpm", "llama"}:
        return preprocess_bunny_with_bos(sources, tokenizer, has_image=has_image)


def preprocess_plain_bunny(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib_bunny.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_bunny(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib_bunny.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib_bunny.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 1
            end_token_cnt += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_bunny_with_bos(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib_bunny.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib_bunny.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        end_token_cnt = 0
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            end_token_cnt += 1
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


# llava preprocess
def preprocess_text_llava(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib_llava.default_conversation.sep_style == conversation_lib_llava.SeparatorStyle.PLAIN:
        return preprocess_plain_llava(sources, tokenizer)
    if conversation_lib_llava.default_conversation.sep_style == conversation_lib_llava.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib_llava.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib_llava.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib_llava.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib_llava.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib_llava.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain_llava(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib_llava.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib_llava.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib_llava.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len