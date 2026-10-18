# export CACHE_ROOT="ROOT PATH"
# export MODEL_NAME="${CACHE_ROOT}/huggingface/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819"

# Loop through the array and set the OUTPUT_DIR variable accordingly
export OUTPUT_DIR="./LoRA/checkpoint/PACS_final_train_ma"
export DATASET_NAME="./PACS_final_train_ma"
export LOG_DIR="./LoRA/train/logs"

echo "script execution. gt_dm"
accelerate launch --mixed_precision="fp16" ./finetune/train_text_to_image_lora.py \
--train_data_dir $DATASET_NAME --caption_column="text" \
--report_to=tensorboard \
--resolution=512 --random_flip \
--train_batch_size=8 \
--num_train_epochs=100 --checkpointing_steps=500 \
--learning_rate=1e-04 --lr_scheduler="constant" \
--seed=42 \
--output_dir=${OUTPUT_DIR} \
--snr_gamma=5 \
--guidance_token=8 \
--dist_match=0.003 \
--logging_dir $LOG_DIR \
--exp_id 0
echo "All processes completed"
