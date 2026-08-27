#!/bin/bash

PORT=25111

CUDA_IDX='0'
NUM_PROC=1
cfg_path=/root/code/configs/mvsec.yaml

save_dir=/root/code/results/mvsec
checkpoint_path=/root/code/results/mvsec/weights/mvsec_model.pth

export CUDA_VISIBLE_DEVICES=${CUDA_IDX} 

torchrun --nproc_per_node=$NUM_PROC --master_port=$PORT ../inference.py \
 --cfg_path ${cfg_path} --save_dir ${save_dir} --checkpoint_path  ${checkpoint_path}