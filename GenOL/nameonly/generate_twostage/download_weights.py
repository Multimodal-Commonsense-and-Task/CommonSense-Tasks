import torch
from transformers import AutoModel, AutoTokenizer

# Auraflow
# from diffusers import AuraFlowPipeline
# pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow-v0.2", torch_dtype=torch.float16, variant="fp16").to("cuda")


# # Floyd
# from diffusers import DiffusionPipeline
# DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
# stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
# stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16)
# safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
# stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)

# # SDXL
# from diffusers import DiffusionPipeline
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
# refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

# # SD3
# from diffusers import StableDiffusion3Pipeline
# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)