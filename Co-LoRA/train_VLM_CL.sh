#!/bin/bash

# CIL CONFIG fedsim_feddualMultipqfullfreeze_homoAgg
NOTE="debug_fedmosaic"
MODE="fedmosaic"
RND_SEED=1

# fed args
SCENARIO=DRAKE_hetero_llava_llama_1B_3B
NUM_ROUNDS=5
NUM_TASKS=4
NUM_CLIENTS=10
MODEL_MAX_LEN=20000
NUM_ITER=94


MEMORY_SIZE=100000
IS_CONTINUAL=False

USE_TASK_VECTOR=True

BATCHSIZE=4
IS_MULTIMODAL=True
ONLINE_T=0.125
ONLINE_DECAY_RATIO=0.99

LR=2e-5
MM_PROJECTOR_LR=5e-5 #3e-4
FINAL_LR=$LR #3e-4
MM_FINAL_LR=$MM_PROJECTOR_LR #3e-4
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="constant" #constant cosine
WARMUP_RATIO=0.1 
DECAY_RATIO=0.9
SAVE_OPTIM=True

# dummy model input - real model used is defined in scenario file
MODEL_NAME="thkim0305/llama3.2_1B_vl"
VERSION="llama3"
MODEL_TYPE="llama3"
BITS=16

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port 29715 \
    --include localhost:0 \
    train_VLM_CL.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --num_clients $NUM_CLIENTS \
    --model_max_length $MODEL_MAX_LEN \
    --num_rounds $NUM_ROUNDS \
    --num_tasks $NUM_TASKS \
    --scenario $SCENARIO \
    --note $NOTE \
    --is_multimodal $IS_MULTIMODAL \
    --memory_size $MEMORY_SIZE \
    --is_streamonly False \
    --online_stream_T $ONLINE_T \
    --online_stream_count_decay_ratio $ONLINE_DECAY_RATIO \
    --is_continual $IS_CONTINUAL \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --num_iter $NUM_ITER \
    --gradient_accumulation_steps $BATCHSIZE \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED \
    --optim $OPT_NAME \
    --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --decay_ratio $DECAY_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size 1 \
    --final_lr $FINAL_LR --mm_final_lr $MM_FINAL_LR \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 2 \
    --lora_enable True \
    --save_optim $SAVE_OPTIM \
    --save_per_step False \
    --use_task_vector $USE_TASK_VECTOR \
    --load_pretrained_lora True \
    --softmax_temp 0.5 \
    --grad_freq 10 \
    --get_prompt True \
    --is_cross_model_series False \
    --output_dir "./results/test/" #> ./nohup/${NOTE}.log 2>&1 &

# --eval_period $EVAL_PERIOD
# lr_scheduler_type
#  --load_checkpoint $LOAD_CHECKPOINT \
    # --lora_r 32 \
    # --lora_alpha 64 \
    # --lora_r 16 \
    # --lora_alpha 32 \
    # --is_multimodal False \