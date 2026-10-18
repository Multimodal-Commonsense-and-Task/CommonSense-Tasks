# #!/bin/bash
# #SBATCH -p suma_a6000
# #SBATCH --job-name=random
# #SBATCH --output=logs/%x_%j.out
# #SBATCH --gres=gpu:1
# #SBATCH --exclude=node37
# ##SBATCH -c 32

# ulimit -u 200000
# source ~/.bashrc
# ml purge
# conda init bash
# conda activate sampling
# export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5

#!/bin/bash
# CIL CONFIG
MODE="sft"
MODEL_ARCH="qwen-8b" # llava bunny_3b bunny_8b
SELECTION_METHOD="fisher"
EMBEDDING="representation"
SETUP="disjoint"
SAMPLE_RATIO=0.25
NUM_ITER=0.25

RND_SEED=1
SCENARIO=2
GPU=2
MASTER_PORT=27510
SAMPLING="bottom"
EMA_RATIO=0.9

RESUME_TRAIN_ROUND=-1
LOAD_CHECKPOINT="/share0/jinyeonkim/LMM-CoresetSelection-CL/client_states_v7_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_infoBatch_scenario16_10000_random0_25_seed1/server_model_round2.pth"

# Information4
START_EMA_UPDATE_STEP=320
MIN_Z_SCORE_THRESHOLD=1.54
KMEANS=1
MUTUAL_PARAM=1
if [ "$SAMPLE_RATIO" == "0.0625" ]; then
    MIN_Z_SCORE_THRESHOLD=1.54
    MIN_Z_SCORE_THRESHOLD2=1.54
    KMEANS=1
elif [ "$SAMPLE_RATIO" == "0.125" ]; then
    MIN_Z_SCORE_THRESHOLD=1.15
    MIN_Z_SCORE_THRESHOLD2=1.54
    KMEANS=2
elif [ "$SAMPLE_RATIO" == "0.25" ]; then
    MIN_Z_SCORE_THRESHOLD=0.67
    MIN_Z_SCORE_THRESHOLD2=1.54
    KMEANS=4
fi


if [ "$SCENARIO" == "1" ]; then
    EVAL_PERIOD=1600000000
    NUM_TASKS=15
elif [ "$SCENARIO" == "2" ]; then
    EVAL_PERIOD=1600000000
    NUM_TASKS=15
elif [ "$SCENARIO" == "3" ]; then
    EVAL_PERIOD=1600000000
    NUM_TASKS=9
elif [ "$SCENARIO" == "4" ]; then
    EVAL_PERIOD=16000000000
    NUM_TASKS=19
fi
# 0.1 -> 1.28, 0.2 -> 0.84, 0.3 -> 0.52

MAX_Z_SCORE_THRESHOLD=5
INFO_UPDATE_MODE="max"
EMA_UPDATE_MODE="none"
EMA_AVERAGE=False
EMA_POOL_SIZE=100


SOFTMAX_UPDATE_TEMP=0.5
# fed args
NUM_ROUNDS=1
NUM_CLIENTS=1
MODEL_MAX_LEN=20000
MEMORY_SIZE=10000000
IS_STREAMONLY=False
IS_SAR=True
IS_SAR_TASK=False

LORA_ENABLE=True
IA3_ENABLE=False

USE_TASK_ID=False
USE_PROMPT=False

SAVE_OPTIM=True

USE_TASK_VECTOR=False
USE_FISHER=False

GENERATOR_OUTPUT_SIZE=1024
GENERATOR_HIDDEN_DIM=8
GENERATOR_HIDDEN_FEATURE=8
KEY_EMBED_SIZE=64
POOL_SIZE=4
PROMPT_TOP_K=1
MODEL_EMA_RATIO=0.99

BATCHSIZE=1

if [ "$MODEL_ARCH" == "llama-8b" ]; then
    MODEL_NAME="meta-llama/Llama-3.1-8B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-4
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "llama-1b" ]; then
    MODEL_NAME="meta-llama/Llama-3.2-1B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-4
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "llama-3b" ]; then
    MODEL_NAME="meta-llama/Llama-3.2-3B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-4
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "qwen-8b" ]; then
    MODEL_NAME="Qwen/Qwen3-8B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-4
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "qwen-4b" ]; then
    MODEL_NAME="Qwen/Qwen3-4B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-4
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "qwen-1b" ]; then
    MODEL_NAME="Qwen/Qwen3-0.6B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-4
    BITS=16
    IS_MULTIMODAL=False
else
    echo "Undefined setting"
    exit 1
fi

NOTE=LLM_${MODEL_ARCH}_${LR}_bs1_gradacc32_${MODE}_scenario${SCENARIO}_random${SAMPLE_RATIO}_seed${RND_SEED}_iter${NUM_ITER}_${SELECTION_METHOD}

MM_PROJECTOR_LR=0 #3e-4
FINAL_LR=$LR #3e-4
MM_FINAL_LR=$LR #3e-4
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine" #cosine
WARMUP_RATIO=0.1 # SHOULD BE 0.03 / NUM_ROUNDS
DECAY_RATIO=0.9

# --master_port 29500
# --num_gpus=4
# LOAD_CHECKPOINT="client_states_fedours_bs4_saveoptim_lr6e-3_lastdownmean_freq5_fishercossimsoftmax_mean_sc0_4tasks_5rounds_fixitr100/server_model_round4.pth"
# LOAD_CHECKPOINT="client_states_fedours_bs4_saveoptim_lr6e-3_lastdownmean_freq5_fishercossimsoftmax_mean_sc0_4tasks_5rounds_fixitr100/server_model_round4.pth"
# # --deepspeed ./deepspeed_script/zero2.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port $MASTER_PORT \
    --include localhost:${GPU} \
    train_LLM_CL.py \
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
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --num_iter $NUM_ITER \
    --gradient_accumulation_steps 16 \
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
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --final_lr $FINAL_LR --mm_final_lr $MM_FINAL_LR \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --save_strategy "no" \
    --logging_steps 2 \
    --note $NOTE \
    --memory_size $MEMORY_SIZE \
    --is_streamonly $IS_STREAMONLY \
    --is_sar $IS_SAR \
    --is_sar_task $IS_SAR_TASK \
    --prompt_num 1 \
    --lora_enable $LORA_ENABLE \
    --ia3_enable $IA3_ENABLE \
    --use_task_id $USE_TASK_ID \
    --get_prompt $USE_PROMPT \
    --generator_output_size $GENERATOR_OUTPUT_SIZE \
    --generator_hidden_dim $GENERATOR_HIDDEN_DIM \
    --generator_hidden_feature $GENERATOR_HIDDEN_FEATURE \
    --ema_ratio $EMA_RATIO \
    --model_ema_ratio $MODEL_EMA_RATIO \
    --key_embed_size $KEY_EMBED_SIZE \
    --pool_size $POOL_SIZE \
    --prompt_top_k $PROMPT_TOP_K \
    --save_optim $SAVE_OPTIM \
    --use_task_vector $USE_TASK_VECTOR \
    --use_fisher $USE_FISHER \
    --fedours False \
    --selection_method $SELECTION_METHOD \
    --embedding $EMBEDDING \
    --k_means $KMEANS \
    --sample_ratio $SAMPLE_RATIO \
    --start_ema_update_step $START_EMA_UPDATE_STEP \
    --min_z_score_threshold $MIN_Z_SCORE_THRESHOLD \
    --min_z_score_threshold2 $MIN_Z_SCORE_THRESHOLD2 \
    --max_z_score_threshold $MAX_Z_SCORE_THRESHOLD \
    --ema_update_mode $EMA_UPDATE_MODE \
    --softmax_update_temp $SOFTMAX_UPDATE_TEMP \
    --sampling $SAMPLING \
    --ema_average $EMA_AVERAGE \
    --ema_pool_size $EMA_POOL_SIZE \
    --info_update_mode $INFO_UPDATE_MODE \
    --setup $SETUP \
    --mutual_param $MUTUAL_PARAM \
    --eval_period $EVAL_PERIOD \
    --resume_train_round $RESUME_TRAIN_ROUND \
    --load_checkpoint $LOAD_CHECKPOINT \
    --is_multimodal $IS_MULTIMODAL \
    --output_dir "./results/test/" #> ./nohup2/${NOTE}.log 2>&1 &

# --selection_method $SELECTION_METHOD \

# --load_checkpoint $LOAD_CHECKPOINT \

# --eval_period $EVAL_PERIOD
# lr_scheduler_type
# --include localhost:7 \