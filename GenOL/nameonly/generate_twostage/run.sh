#!/bin/bash

# Datacenter SBATCH parameters
#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
##SBATCH -q big_qos
#SBATCH --job-name=50_2_I_floyd
#SBATCH --output=logs/%x_%j.out
##SBATCH --exclude=node37

# # Uncomment when running on datacenter
# source ~/.bashrc
# ml purge
# conda init bash
# conda activate generate # cogview

# ----------------- IMPORTANT -----------------
DATASET="ImageNet" # PACS, DomainNet, cifar10, NICO
IMAGE_DIR='./generated_datasets/ImageNet_fake_f_cogview2_more'
GENERATIVE_MODEL="cogview2" # sdxl, floyd, cogview2, sd3, sdturbo, flux, kolors, auraflow
# WARNING: Do not split the class indices across multiple runs in the same server
START_CLASS=0
END_CLASS=999
PROMPT_DIR='../prompt_generation/prompts/fake_f_ImageNet.json'
INCREASE_RATIO=1.15
# Ignored when running on datacenter
GPU_ID=${1:-0}
LORA_PATH="none"
# ----------------- IMPORTANT -----------------

# Uncomment when running on OUR gpu servers (both cogview2 and others supported)
BASENAME=$(basename $IMAGE_DIR)
mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python get_image_queue.py --config_path ./configs/default.yaml --dataset $DATASET \
--image_dir $IMAGE_DIR --generative_model $GENERATIVE_MODEL --start_class $START_CLASS --end_class $END_CLASS \
--prompt_dir $PROMPT_DIR --increase_ratio $INCREASE_RATIO --lora_path $LORA_PATH \
 > "logs/${BASENAME}_${GPU_ID}.log" 2>&1 &


# # Uncomment following lines when running on datacenter
# if [ "$GENERATIVE_MODEL" == "cogview2" ]; then
#     echo "Running on cogview2"
#     cd Image-Local-Attention
#     CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python setup.py install
#     cd ../
#     CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python get_image_queue.py --config_path ./configs/default.yaml --dataset $DATASET --image_dir $IMAGE_DIR \
#     --generative_model $GENERATIVE_MODEL --start_class $START_CLASS --end_class $END_CLASS --prompt_dir $PROMPT_DIR \
#     --increase_ratio $INCREASE_RATIO --lora_path $LORA_PATH
# else
#     python get_image_queue.py --config_path ./configs/default.yaml --dataset $DATASET --image_dir $IMAGE_DIR \
#     --generative_model $GENERATIVE_MODEL --start_class $START_CLASS --end_class $END_CLASS --prompt_dir $PROMPT_DIR \
#     --increase_ratio $INCREASE_RATIO --lora_path $LORA_PATH
# fi

# Estimated time for image generation
# 4090
# Floyd: 2.2857/min, 0.4375min/1 image ? 2/min??
# SDXL: 11/min, 0.0909min/1 image
# sdturbo: fast
# cogview2: 4/min, 0.25min/1 image
# flux: 4/min, 0.25min/1 image
# kolors: 8.57/min, 0.1166min/1 image
# auraflow: 1.5/min, 0.6667min/1
