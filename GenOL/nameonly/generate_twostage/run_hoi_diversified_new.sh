#!/bin/bash

# Datacenter SBATCH parameters
#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH -q big_qos
#SBATCH --job-name=hoi_LE_floyd
#SBATCH --output=logs/%x_%j.out
##SBATCH --exclude=node37

## Uncomment when running on datacenter
# source ~/.bashrc
# ml purge
# conda init bash
# conda activate generate # cogview

# ----------------- IMPORTANT -----------------
IMAGE_DIR='./generated_datasets/generated_LE_ver1_floyd'
GENERATIVE_MODEL="floyd" # sdxl, floyd, cogview2, sd3, sdturbo, flux, kolors, auraflow
START_CLASS=0
END_CLASS=3603 # 3603
PROMPT_DIR='../prompt_generation/prompts/generated_LE_ver1.json'
GPU_ID=0

# Uncomment when running on OUR gpu servers (both cogview2 and others supported)
BASENAME=$(basename $IMAGE_DIR)
mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python generate_images_hoi_diversified_new.py -m $GENERATIVE_MODEL \
-r $IMAGE_DIR -s $START_CLASS -e $END_CLASS -p $PROMPT_DIR > "logs/${BASENAME}_${GPU_ID}.log" 2>&1 &

# # Uncomment following lines when running on datacenter
# if [ "$GENERATIVE_MODEL" == "cogview2" ]; then
#     echo "Running on cogview2"
#     cd Image-Local-Attention
#     CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python setup.py install
#     cd ../
#     CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python generate_images_hoi_diversified_new.py -m $GENERATIVE_MODEL \
#     -r $IMAGE_DIR -s $START_CLASS -e $END_CLASS -p $PROMPT_DIR
# else
#     python generate_images_hoi_diversified_new.py -m $GENERATIVE_MODEL -r $IMAGE_DIR -s $START_CLASS \
#     -e $END_CLASS -p $PROMPT_DIR
# fi

# Estimated time for image generation
# 4090
# Floyd: 2.2857/min, 0.4375min/1 image ? 2/min??
# SDXL: 11/min, 0.0909min/1 image
# sdturbo: fast
# cogview2: 4/min, 0.25min/1 image
# flux: 4/min, 0.25min/1 image
# kolors: 8.57/min, 0.1166min/1 image
# auraflow: 1.5/min, 0.6667min/1 image