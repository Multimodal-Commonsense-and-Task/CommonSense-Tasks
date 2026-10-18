import os
import json
import re
import argparse
import numpy as np
import torch
import pickle
import random
import requests
import yaml
import json
import socket
import shutil
from PIL import Image
from classes import *
from io import BytesIO
from parse_prompt import get_class_prompt_dict
from tqdm import tqdm

gpu_number = os.environ.get('CUDA_VISIBLE_DEVICES', 'None')
print(f"Using GPU: {gpu_number}")

n_steps = 40
high_noise_frac = 0.8

def find_free_port():
    """Find a free port for distributed initialization."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
    
def get_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_three_numbers(file_name):
    parts = file_name.split('_')
    if len(parts) >= 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            # Ignore if the file name does not contain three numbers
            return None
    elif len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1]), 0
        except ValueError:
            # Ignore if the file name does not contain two numbers
            return None
    return None

class ImageGenerator:
    def __init__(self, generative_model):
        self.generative_model = generative_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        raise NotImplementedError("load_model method is not implemented")
    
    def generate_one_image(self, prompt):
        raise NotImplementedError("generate_one_image method is not implemented")

class SDXLGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("sdxl")
        self.load_model()
    
    def load_model(self):
        from diffusers import DiffusionPipeline
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                        variant="fp16", use_safetensors=True).to(self.device)
        self.refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=self.pipe.text_encoder_2,
                                                        vae=self.pipe.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)

    def generate_one_image(self, prompt):
        image = self.pipe(prompt=prompt, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type="latent").images
        image = self.refiner(prompt=prompt, num_inference_steps=n_steps, denoising_start=high_noise_frac, image=image).images[0]
        return image


class SD3Generator(ImageGenerator):
    def __init__(self):
        super().__init__("sd3")
        self.load_model()
    
    def load_model(self):
        from diffusers import StableDiffusion3Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(self.device)

    def generate_one_image(self, prompt):
        image = self.pipe(prompt=prompt, negative_prompt="", num_inference_steps=28, height=1024, width=1024, guidance_scale=7.0).images[0]
        return image
    
class Kandinsky2Generator(ImageGenerator):
    def __init__(self, decoder_steps=50, batch_size=1, h=256, w=256):
        super().__init__("kandinsky2")
        self.load_model()
        self.decoder_steps = decoder_steps
        self.batch_size = batch_size
        self.h = h
        self.w = w
        assert self.batch_size == 1, "Currently only batch_size=1 is supported"

    def load_model(self):
        from kandinsky2 import get_kandinsky2
        self.model = get_kandinsky2(self.device, task_type='text2img', model_version='2.2')
    
    def generate_one_image(self, prompt):
        image = self.model.generate_text2img(prompt, decoder_steps=self.decoder_steps, batch_size=self.batch_size, h=self.h, w=self.w)
        return image[0]

class KarloGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("karlo")
        self.load_model()
    
    def load_model(self):
        from diffusers import UnCLIPPipeline
        self.pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16).to(self.device)

    def generate_one_image(self, prompt):
        image = self.pipe([prompt]).images[0]
        return image

class SDTurboGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("sdturbo")
        self.load_model()
    
    def load_model(self):
        from diffusers import AutoPipelineForText2Image
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(self.device)
    
    def generate_one_image(self, prompt):
        image = self.pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
        return image

class FluxGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("flux")
        self.load_model()
    
    def load_model(self):
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.to(torch.float16)

    def generate_one_image(self, prompt):
        image = self.pipe(prompt=prompt, guidance_scale=0, num_inference_steps=4, max_sequence_length=256).images[0]
        return image

class KolorsGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("kolors")
        self.load_model()
    
    def load_model(self):
        from diffusers import KolorsPipeline
        self.pipe = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16").to("cuda")

    def generate_one_image(self, prompt):
        image = self.pipe(prompt=prompt, negative_prompt="", guidance_scale=5.0, num_inference_steps=50).images[0]
        return image

class AuraFlowGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("kolors")
        self.load_model()
    
    def load_model(self):
        from diffusers import AuraFlowPipeline
        self.pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow-v0.2", torch_dtype=torch.float16, variant="fp16").to("cuda")

    def generate_one_image(self, prompt):
        image = self.pipe(prompt=prompt, guidance_scale=3.5, num_inference_steps=50, height=1024, width=1024).images[0]
        return image

class FloydGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("floyd")
        self.load_model()
        print(f"FloydGenerator model requires huggingface login to download the model.")
        print(f"Make sure you have logged in to huggingface using `from huggingface_hub import login; login()`")
    
    def load_model(self):
        from diffusers import DiffusionPipeline
        print(f"Loading Floyd model - stage 1")
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        # stage_1.enable_xformers_memory_efficient_attention() # remove line if torch.__version__ >= 2.0.0
        stage_1.enable_model_cpu_offload()
        
        print(f"Loading Floyd model - stage 2")
        stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
        # stage_2.enable_xformers_memory_efficient_attention() # remove line if torch.__version__ >= 2.0.0
        stage_2.enable_model_cpu_offload()

        print(f"Loading Floyd model - stage 3")
        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
        # stage_3.enable_xformers_memory_efficient_attention() # remove line if torch.__version__ >= 2.0.0
        stage_3.enable_model_cpu_offload()

        self.stage_1 = stage_1; self.stage_2 = stage_2; self.stage_3 = stage_3

    def generate_one_image(self, prompt):
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)
        image = self.stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images
        image = self.stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images
        image = self.stage_3(prompt=prompt, image=image, noise_level=100).images
        image = image[0]
        return image

def adjust_list_length(lst, length, resume_prompt_idx=None):
    if resume_prompt_idx is not None:
        print(f"Start generating from the prompt index {resume_prompt_idx}")
        lst = lst[resume_prompt_idx:] + lst[:resume_prompt_idx]
    
    original_length = len(lst)
    
    if length <= original_length:
        return lst[:length]
    else:
        repeated_list = lst * (length // original_length)
        remaining_part = random.sample(lst, length % original_length)
    return repeated_list + remaining_part

class CogView2Generator(ImageGenerator):
    def __init__(self): 
        super().__init__("CogView2")
        self.load_model()
    
    def load_model(self):
        from CogView2.generator import Cogview2
        self.model = Cogview2(img_size=224, style='photo', batch_size=1, max_inference_batch_size=1)
    
    def generate_one_image(self, prompt):
        image = self.model.generate_images(prompt)
        return image

def model_selector(generative_model, API_KEY=None):
    if generative_model == "sdxl":
        return SDXLGenerator()
    elif generative_model == "sd3":
        return SD3Generator()
    elif generative_model == "kandinsky2":
        return Kandinsky2Generator()
    elif generative_model =="floyd":
        return FloydGenerator()
    elif generative_model == "cogview2":
        # Set port number
        free_port = find_free_port()
        print(f"Set MASTER_PORT to {free_port}!")
        os.environ["MASTER_PORT"] = str(free_port)
        return CogView2Generator()
    elif generative_model == "dalle2":
        return DALLE2Generator(api_key=API_KEY)
    elif generative_model == "karlo":
        return KarloGenerator()
    elif generative_model == "sdturbo":
        return SDTurboGenerator()
    elif generative_model == "flux":
        return FluxGenerator()
    elif generative_model == "kolors":
        return KolorsGenerator()
    elif generative_model == "auraflow":
        return AuraFlowGenerator()
    else:
        raise ValueError(f"Generative model {generative_model} is not supported")

class DALLE2Generator(ImageGenerator):
    def __init__(self, api_key):
        super().__init__("DALLE2")
        from openai import OpenAI
        self.api_key = api_key
        self.client = OpenAI(api_key = self.api_key)
        
    def download_image(self, url):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def generate_one_image(self, prompt):
        response = self.client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="256x256",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        image = self.download_image(image_url)

        return image

    def generate_10_images(self, prompt):
        response = self.client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="256x256",
            quality="standard",
            n=10,
        )
        image_urls = [img.url for img in response.data]
        images = [self.download_image(url) for url in image_urls]

        # Remove None images
        images = [img for img in images if img is not None]
        return images
    
    def generate_k_images(self, prompt, k):
        try:
            response = self.client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="256x256",
                quality="standard",
                n=k,
            )
        except Exception as e:
            print(f"Error generating images: {e}")
            return None
        image_urls = [img.url for img in response.data]
        images = [self.download_image(url) for url in image_urls]

        # Remove None images
        images = [img for img in images if img is not None]
        return images
    
def get_image(prompt, model):
    image = model.generate_one_image(prompt)
    return image

def generate_unique_filename(directory, base_filename):
    filename, extension = os.path.splitext(base_filename)
    counter = 1

    new_filename = base_filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{filename}_{counter}{extension}"
        counter += 1
    return new_filename

def generate_single_class(
    class_name,
    model_name,
    model,
    image_dir,
    num_samples,
    class_prompt_dict,
    result_json_path,
    API_KEY=None,
    resume_prompt_idx=None
):
    if not 'metaprompts' in class_prompt_dict:
        use_dynamic_prompt = True
        class_prompt_dict = class_prompt_dict[class_name]
    else:
        use_dynamic_prompt = False
    
    os.makedirs(os.path.join(image_dir), exist_ok=True)
    if class_num_samples_dict is not None:
        class_list = list(class_num_samples_dict.keys())
        print(f"Set class_list as {class_list}")
    
    print(f"Generating images for class {class_name} ({num_samples} images)")
    # Find the maximum prompt index already generated
    if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
        images = os.listdir(image_dir)
        if len(images) == num_samples:
            print(f"Images already exist in {image_dir} ({len(images)} images)")
            return
        
        print(f"Images already exist in {image_dir} ({len(images)} images)")
        image_prefixes = [image.split('.')[0] for image in images]
        image_prefixes = [extract_three_numbers(prefix) for prefix in image_prefixes] # (metaprompt_idx, diversified_prompt_idx, iter)
        image_prefixes = [prefix for prefix in image_prefixes if prefix is not None]
        
        # Find the maximum iteration
        max_iter = max([prefix[2] for prefix in image_prefixes])
        image_prefixes = [(prefix[0], prefix[1], prefix[2]) for prefix in image_prefixes if prefix[2] == max_iter]
        max_prompt_indices = sorted(image_prefixes, reverse=True)[0]
        print(f"Maximum prompt index - metaprompt: {max_prompt_indices[0]}, diversified: {max_prompt_indices[1]}, iter: {max_prompt_indices[2]}")
        
        # Find the number of samples to generate
        num_samples = num_samples - len(os.listdir(image_dir))
        print(f"Set the number of samples to generate to {num_samples}")
    else:
        max_prompt_indices = None
    
    # First adjust the number of iteration to generate enough images
    # 1. concat metaprompt and diversified prompts and label each index
    concatenated_prompt_list = []
    for metaprompt_dict in class_prompt_dict['metaprompts']:
        metaprompt_idx = metaprompt_dict['index']
        for prompt_dict in metaprompt_dict['prompts']:
            prompt_info = (prompt_dict['content'], 'diversified', f"{metaprompt_idx}_{prompt_dict['index']}")
            concatenated_prompt_list.append(prompt_info)

    # Replace the placeholder [concept] into class name
    for i, prompt in enumerate(concatenated_prompt_list):
        if not use_dynamic_prompt:
            assert "[concept]" in prompt[0], f"[concept] not exists in {prompt}!"
            prompt_with_cls = (prompt[0].replace('[concept]', class_name), prompt[1], prompt[2])
            concatenated_prompt_list[i] = prompt_with_cls
        else:
            concatenated_prompt_list[i] = (prompt[0], prompt[1], prompt[2])
    
    # Find the prompt index to start generating images
    if max_prompt_indices is not None:
        resume_prompt_idx = next(i for i, v in enumerate(concatenated_prompt_list) if v[2] == f"{max_prompt_indices[0]}_{max_prompt_indices[1]}")
        resume_prompt_idx += 1
    
    concatenated_prompt_list = adjust_list_length(concatenated_prompt_list, num_samples, resume_prompt_idx=resume_prompt_idx)

    # Dictionary to store image info (metaprompt, diversified prompt)
    info_dict = {}

    # Generate images according to the adjusted prompt
    for i, (prompt, prompt_type, image_name) in enumerate(tqdm(concatenated_prompt_list)):
        print(f"Generating image for {prompt_type} - {image_name} - {prompt}")

        while True:
            try:
                image = get_image(prompt, model)
                if image is not None:
                    break
            except Exception as e:
                print(e)
                continue
            
        image = image.resize((224, 224))

        unique_image_name = generate_unique_filename(image_dir, image_name + ".jpg")
        info_dict[unique_image_name] = {'type': prompt_type, 'prompt': prompt}
        image_path = os.path.join(image_dir, unique_image_name)
        image.save(image_path, "JPEG")

    with open(result_json_path, 'w') as f:
        json.dump(info_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./configs/default.yaml')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--generative_model', type=str, default=None)
    parser.add_argument('--start_class', type=int, default=None)
    parser.add_argument('--end_class', type=int, default=None)
    parser.add_argument('--prompt_dir', type=str, default=None)
    parser.add_argument('--increase_ratio', type=float, default=None)
    
    parser_config = parser.parse_args()

    config = get_config(config_path=parser_config.config_path)
    
    # Override config if arguments are explicitly provided (start, end class)
    if parser_config.dataset is not None:
        config['dataset'] = parser_config.dataset
    if parser_config.image_dir is not None:
        config['image_dir'] = parser_config.image_dir
    if parser_config.generative_model is not None:
        config['generative_model'] = parser_config.generative_model
    if parser_config.start_class is not None:
        config['start_class'] = parser_config.start_class
        config['end_class'] = parser_config.end_class
    if parser_config.prompt_dir is not None:
        config['prompt_dir'] = parser_config.prompt_dir
    if parser_config.increase_ratio is not None:
        config['increase_ratio'] = parser_config.increase_ratio
    
    print(f"Config: {config}")
    
    model_name = config['generative_model']
    image_root_dir = config['image_dir']
    debug = config['debug']

    sample_num_dict = count_dict[config['dataset']]
    classes = list(sample_num_dict.keys())
    if 'end_class' in config:
        classes = classes[config['start_class']:config['end_class'] + 1]
        print(f"Set classes to {classes}")
        print(f"Class indices: {config['start_class']} ~ {config['end_class']}")
    
    # Load prompt file  
    with open(config['prompt_dir'], 'r') as f:
        class_prompt_dict = json.load(f)

    # Set the number of samples (follow yaml if provided, else dataset_dict.pkl)
    if config.get('num_samples_per_cls'):
        print(f"WARNING: Set the number of all classes to {config['num_samples_per_cls']}")
        class_num_samples_dict = {cls: config['num_samples_per_cls'] for cls in classes}
    elif config.get('num_samples'):
        print(f"WARNING: The number of samples of each class are manually provided")
        classes = config['num_samples'].keys()
        class_num_samples_dict = config['num_samples']
    else:
        print(f"Generate the default (MA size * increase ratio ({config['increase_ratio']}))")
        class_num_samples_dict = {cls: int(num * config['increase_ratio']) for cls, num in sample_num_dict.items()}
        class_num_samples_dict = {cls: class_num_samples_dict[cls] for cls in classes}

    print(f"IMPORTANT: number of samples to generate for each class")
    print(class_num_samples_dict)

    # Load model
    if debug:
        model = None
    else:
        model = model_selector(config['generative_model'], API_KEY=config['api_key'])

    # Set start prompt index if provided
    resume_prompt_idx = config.get('resume_prompt_idx')
    
    # Generate each class
    for cls in tqdm(classes):
        print(f"Start generating class {cls}...")
        num_samples_cls = class_num_samples_dict[cls]

        generate_single_class(
            class_name=cls,
            model_name=config['generative_model'],
            model=model,
            image_dir = os.path.join(image_root_dir, cls),
            num_samples = num_samples_cls,
            class_prompt_dict=class_prompt_dict,
            API_KEY=config['api_key'],
            result_json_path=os.path.join(image_root_dir, f"{cls}.json"),
            resume_prompt_idx=resume_prompt_idx
        )