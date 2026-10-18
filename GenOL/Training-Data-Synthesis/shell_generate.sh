#!/bin/bash

DATASET_NAME
lora_path="./LoRA/checkpoint/PACS_final_synclr"
metadata_path="PACS_final_synclr/metadata.jsonl"
guidance_tokens=('Yes')
target_path="PACS_final_real_fake"
SDXLs=('No')
image_strengths=(0.75)

guidance_token="${guidance_tokens[$i]}"
SDXL="${SDXLs[$i]}"
imst="${image_strengths[$i]}"
echo "$ver LoRA: $lora Method $method"
# Iterate from 0-7, cover all case for nchunks <= 8

j=3
echo $j
CUDA_VISIBLE_DEVICES=$j python generate.py --index ${j} --method "SDT2I_LoRA" --batch_size 24 \
--use_caption "blip2" --lora_path $lora_path --if_SDXL $SDXL --use_guidance $guidance_token \
--img_size 512 --cross_attention_scale 0.5 --image_strength $imst --nchunks 1 \
--metadata_path $metadata_path --target_path $target_path
