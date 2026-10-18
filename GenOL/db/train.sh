# #!/bin/bash

IMAGE_DIR="PACS_final_train_ma_5"

# Train dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="lora-trained-xl_PACS_final_train_ma_5"
# export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --dataset_name=$IMAGE_DIR \
  --caption_column="caption" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" \
  --push_to_hub \
  --instance_prompt="temp" \
  # --with_prior_preservation \
  # --class_data_dir=$CLASS_DATA_DIR \
  # --class_prompt="$CLASS_PROMPT" \