import os
from pathlib import Path
import re
import shutil
import argparse
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.utils import load_image

def load_caption_dict(image_names,caption_path):
    class_lis = set([image_name.split("/")[0] for image_name in image_names])
    dict_lis = []
    for class_name in class_lis:
        with open(os.path.join(caption_path,f"{class_name}.json"), 'r') as file:
            dict_lis.append(json.load(file))
    caption_dict = {key: value for dictionary in dict_lis for key, value in dictionary.items()}
    return caption_dict

def group_lists(list1, list2, list3, list4, list5):
    grouped_data = {}
    for idx, item in enumerate(list1):
        if item not in grouped_data:
            grouped_data[item] = ([list2[idx]], [list3[idx]], [list4[idx]], [list5[idx]])
        else:
            grouped_data[item][0].append(list2[idx])
            grouped_data[item][1].append(list3[idx])
            grouped_data[item][2].append(list4[idx])
            grouped_data[item][3].append(list5[idx])

    grouped_list = [(key, grouped_data[key][0], grouped_data[key][1], grouped_data[key][2], grouped_data[key][3]) for key in grouped_data]
    return grouped_list
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="imagenette", help="Which Dataset")
    parser.add_argument("--start_index", required=True, type=int,help="split task")
    parser.add_argument("--end_index", required=True, type=int,help="split task")
    parser.add_argument("--version",default="v57",type=str,help="out_version")
    parser.add_argument("--lora_path",default=None,type=str,help="lora path")
    parser.add_argument("--batch_size",default=8,type=int,help="batch size")
    parser.add_argument('--use_caption', default='blip2', type=str, help="use caption model")
    parser.add_argument('--img_size',default=512,type=int, help='Generation Image Size')
    parser.add_argument('--target_path', type=str, required=True, help="target path")

    parser.add_argument('--method', default='SD_T2I', type=str, help="generation method")
    parser.add_argument('--use_guidance', default='No', type=str, help="guidance token")
    parser.add_argument('--if_SDXL',default='No', type=str, help="SDXL")
    parser.add_argument('--if_full',default='Yes',type=str, help='singleLora')
    parser.add_argument('--if_compile',default='No',type=str, help='compile?')
    parser.add_argument('--image_strength',default=0.75,type=float,help="init image strength")
    parser.add_argument('--nchunks',default=8,type=int,help="No. subprocess")
    parser.add_argument('--metadata_path',default=None,type=str,help="metadata path")
    parser.add_argument("--imagenet_path",default="./temp",type=str,help="path to imagenet")
    parser.add_argument("--syn_path",default="",type=str,help="path to synthetic data")
    
    # Parameters
    parser.add_argument('--cross_attention_scale', default=0.5, type=float, help="lora scale")
    parser.add_argument('--ref_version',default='v120',type=str, help='version to refine')
    
    args = parser.parse_args()
    return args

class StableDiffusionHandler:
    def __init__(self, args):
        self.args = args
        """
        (Pdb) print(self.args)
        Namespace(batch_size=24, cross_attention_scale=0.5, dataset='imagenette', if_SDXL='No', if_compile='No', if_full='Yes', image_strength=0.75, img_size=512, index=0, lora_path='./LoRA/checkpoint/gt_dm_v1', method='SDI2I_LoRA', nchunks=8, ref_version='v120', use_caption='blip2', use_guidance='Yes', version='v1')
        """
        self.method = args.method   # SDI2I_LoRA
        self.if_SDXL = False
        self.use_guidance_tokens = True
        self.if_full = True 
        self.if_compile = False
        self.metadata_path = args.metadata_path
        self.controlnet_scale = 1.0
        self.lora_path = args.lora_path 
        self.target_path = args.target_path
        self.batch_size = args.batch_size
        self.inference_step = 30
        self.guidance_scale = 2.0
        self.cross_attention_scale = args.cross_attention_scale  # 0.5
        self.init_image_strength = args.image_strength  # 0.75
        self.scheduler = "UniPC"
        self.img_size = args.img_size   # 512
        self.start_index = args.start_index
        self.end_index = args.end_index
        
    ### Get Pipelines
    def get_stablediffusion(self, stablediffusion_path, lora=None):
        pipe = StableDiffusionPipeline.from_pretrained(
            stablediffusion_path, safety_checker=None, torch_dtype=torch.float16, add_watermarker=False
        )
        if lora:
            print("Load LoRA:", lora)
            pipe.unet.load_attn_procs(lora)

        pipe = self.set_scheduler(pipe)
        pipe.to("cuda")
        if self.if_compile:
            print("Compile UNet")
            torch._dynamo.config.verbose = True
            pipe.unet = torch.compile(pipe.unet)
        pipe.enable_model_cpu_offload()
        return pipe
    
    def set_scheduler(self, pipe):
        if self.scheduler == "UniPC":   #! scheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif self.scheduler == "DPM++2MKarras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
        elif self.scheduler == "DPM++2MAKarras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True, algorithm_type="sde-dpmsolver++")
        return pipe


    ### Generate
    def generate_sd(self,prompts,negative_prompts):
        images = self.pipe(prompts, 
            num_inference_steps=self.inference_step,
            negative_prompt=negative_prompts,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale=self.guidance_scale
            ).images
        
        return images
    
    def generate_sd_lora(self,prompts,negative_prompts, image_names, prev_class_id):
        class_ids = [image_name.split("/")[0] for image_name in image_names]
        groups = group_lists(class_ids, prompts, negative_prompts, negative_prompts, image_names)
        print("Group:",len(groups))
        images = []
        for group in groups:
            class_id, prompts, negative_prompts, _, img_names = group
            if not class_id == prev_class_id and not self.if_full:
                self.pipe = self.get_stablediffusion(class_id)
            if self.use_guidance_tokens:
                guidance_tokens = self.get_guidance_tokens_v2(class_id, img_names)
            else:
                guidance_tokens = None

            sub_images = self.pipe(prompts,
                num_inference_steps=self.inference_step,
                negative_prompt=negative_prompts,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                guidance_scale=self.guidance_scale,
                cross_attention_kwargs={"scale": self.cross_attention_scale},
                # guidance_tokens = guidance_tokens
                ).images
            images.extend(sub_images)
        return images, class_id

    def generate_img2img(self,prompts,init_images,negative_prompts):
        images = self.pipe(prompts,
            num_inference_steps=self.inference_step,
            image=init_images,
            negative_prompt=negative_prompts,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale=self.guidance_scale,
            strength=self.init_image_strength
            ).images
        
        return images
    
    def get_guidance_tokens_v2(self, batch_metadata):
        sampled_files = [metadata['img_features'] for metadata in batch_metadata]
        feature_dist_samples = [torch.load(f) for f in sampled_files]
        guidance_tokens = torch.stack(feature_dist_samples, dim=1).view(len(sampled_files),1,-1) # [bsz, 1, 768]
        return guidance_tokens
        
    ### Dataset Generation Pipe
    def generate_ImageNet1k(self):
        print(f"Generation: {self.method}")
        print("Full",self.if_full)
        self.pipe = self.get_stablediffusion("stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.pipe.to("cuda")
        # Load metadata jsonl file
        with open(f"{self.metadata_path}", "r") as f:
            metadata = [json.loads(line) for line in f]
        
        # Load Prompts
        class_names = [m["file_name"].split("/")[0] for m in metadata]
        base_prompts = [f"photo of {c}" for c in class_names]
        caption_suffix = [m["text"] for m in metadata]
        prompts = [f"{base_prompts[n]}, {caption_suffix[n]}, best quality" for n in range(len(metadata))]
        bs = self.batch_size; img_size = (512, 512)
        
        # Fine start, end cls index
        classes = sorted(list(set(class_names)))
        class_range = list(range(self.start_index, self.end_index + 1))
        selected_classes = [classes[i] for i in class_range]
        print(f"Slicing classes: {selected_classes}")
        metadata_idx = [idx for idx, m in enumerate(metadata) if m["file_name"].split("/")[0] in selected_classes]
        metadata = [metadata[idx] for idx in metadata_idx]
        prompts = [prompts[idx] for idx in metadata_idx]
        print(f"Length of metadata: {len(metadata)}")
        
        # Iterate through batches
        image_idx = 0
        previously_generated_class = None
        for i in range(0, len(metadata), bs):
            batch_metadata = metadata[i:i+bs]
            batch_prompts = prompts[i:i+bs]
            batch_image_names = [m["file_name"].split("/")[1] for m in batch_metadata]
            batch_class_ids = [m["file_name"].split("/")[0] for m in batch_metadata]
            if previously_generated_class is None or previously_generated_class != batch_class_ids[0]:
                lora_path = os.path.join(self.lora_path, f"lora_layers_{batch_class_ids[0]}.pt")
                print(f"Loading LoRA: {lora_path}")
                self.pipe.unet.load_attn_procs(lora_path)
            batch_negative_prompts = ["distorted, unrealistic, blurry, out of frame, cropped, deformed" for n in range(len(batch_metadata))]
            
            print(f"Generation {len(batch_metadata)} images")
            print(f"Class ids: {batch_class_ids}")
            guidance_tokens = self.get_guidance_tokens_v2(batch_metadata)
            
            try:
                sub_images = self.pipe(batch_prompts,
                    num_inference_steps=self.inference_step,
                    negative_prompt=batch_negative_prompts,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    guidance_scale=self.guidance_scale,
                    cross_attention_kwargs={"scale": self.cross_attention_scale},
                    # guidance_tokens = guidance_tokens
                    ).images
            except Exception as e:
                print("Failed to generate images")
                print(e)
                continue
            
            for idx, image in enumerate(sub_images):
                # Resize Image
                image = image.resize((224, 224))
                cls_path = os.path.join(self.target_path, batch_class_ids[idx])
                if not os.path.exists(cls_path):
                    os.makedirs(cls_path)
                    
                # Reset image_idx if new class
                if previously_generated_class is None or previously_generated_class != batch_class_ids[idx]:
                    image_idx = 0
                print(f"Saving Image {image_idx} - image metadata: {batch_metadata[idx]}")
                image_path = os.path.join(cls_path, f"{str(image_idx).zfill(5)}.png")
                image.save(image_path)
                image_idx += 1
                previously_generated_class = batch_class_ids[idx]


        
def main():
    args = get_args()
    # import pdb; pdb.set_trace()
    handler = StableDiffusionHandler(args)
    handler.generate_ImageNet1k()
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()