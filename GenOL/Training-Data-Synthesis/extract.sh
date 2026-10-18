#!/bin/bash
DATASET_NAME="PACS"
GPU_ID=0
DATA_DIR="PACS_final_train_ma"


IMAGE_DIR="./${DATA_DIR}"
FEATURE_PATH="LoRA/ImageNet1K_CLIPEmbedding/VIT_L/${DATA_DIR}"
lora_path="./LoRA/checkpoint/${DATA_DIR}"
metadata_path="${DATA_DIR}/metadata.jsonl"
TARGET_DIR="${DATA_DIR}_real_fake"

# # Generate caption
# CUDA_VISIBLE_DEVICES=$GPU_ID python blip_captioning.py --image_dir $IMAGE_DIR

# # Extract features
# CUDA_VISIBLE_DEVICES=$GPU_ID python extract_feature.py --index 0 --imagenet_path $IMAGE_DIR

# # Append features to jsonl file
# python append_features.py --image_dir $IMAGE_DIR --feature_path $FEATURE_PATH

# # LORA training
# export OUTPUT_DIR="./LoRA/checkpoint/${IMAGE_DIR}"
# export LOG_DIR="./LoRA/train/logs"
# CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --mixed_precision="fp16" ./finetune/train_text_to_image_lora.py \
# --train_data_dir $IMAGE_DIR --caption_column="text" \
# --report_to=tensorboard \
# --resolution=512 --random_flip \
# --train_batch_size=8 \
# --num_train_epochs=10 --checkpointing_steps=500 \
# --learning_rate=1e-04 --lr_scheduler="constant" \
# --seed=42 \
# --output_dir=${OUTPUT_DIR} \
# --snr_gamma=5 \
# --guidance_token=8 \
# --dist_match=0.003 \
# --logging_dir $LOG_DIR \
# --exp_id 0
# echo "All processes completed"


# Generation
guidance_tokens=('Yes')
SDXLs=('No')
image_strengths=(0.75)

guidance_token="${guidance_tokens[$i]}"
SDXL="${SDXLs[$i]}"
imst="${image_strengths[$i]}"
echo "$ver LoRA: $lora Method $method"
# Iterate from 0-7, cover all case for nchunks <= 8

# GPUS=("0" "1" "2" "3" "4" "5")
# nchunks=${#GPUS[@]}
# mkdir -p ./log
# for ((j=0; j<${#GPUS[@]}; j++)); do
#     CUDA_VISIBLE_DEVICES=${GPUS[j]} nohup python generate.py \
#         --index ${j} \
#         --method "SDT2I_LoRA" \
#         --batch_size 24 \
#         --use_caption "blip2" \
#         --lora_path $lora_path \
#         --if_SDXL $SDXL \
#         --use_guidance $guidance_token \
#         --img_size 512 \
#         --cross_attention_scale 0.5 \
#         --image_strength $imst \
#         --nchunks $nchunks \
#         --metadata_path $metadata_path \
#         --target_path $TARGET_DIR > ./log/${DATA_DIR}_real_fake_${j}.log 2>&1 &
# done

GPU_ID=5
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python generate.py --start_index 6 --end_index 6 --method "SDT2I_LoRA" --batch_size 24 \
--use_caption "blip2" --lora_path $lora_path --if_SDXL $SDXL --use_guidance $guidance_token \
--img_size 512 --cross_attention_scale 0.5 --image_strength $imst --nchunks 3 \
--metadata_path $metadata_path --target_path $TARGET_DIR > ./log/${DATA_DIR}_real_fake_${GPU_ID}.log 2>&1 &