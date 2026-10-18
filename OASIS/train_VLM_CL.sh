mkdir -p importance
mkdir -p nohup

#!/bin/bash
# CIL CONFIG
MODE="${MODE:-sft}"
MODEL_ARCH="${MODEL_ARCH:-llava}" # llava qwen internvl3-9b
RND_SEED=1
SELECTION_METHOD="${SELECTION_METHOD:-oasis}"
EMBEDDING="representation"
SETUP="disjoint"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.0625}"

SCENARIO=20
GRAD_RATIO=1 #0.25 0.125 0.0125 0.00625 0.003125

GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-28005}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
SAMPLING="bottom"
EMA_RATIO=0.9


RESUME_TRAIN_ROUND="${RESUME_TRAIN_ROUND:--1}"
# Information
START_EMA_UPDATE_STEP=0
KMEANS=1
MUTUAL_PARAM=1
if [ "$SAMPLE_RATIO" == "0.0625" ]; then
    DEFAULT_MIN_Z_SCORE_THRESHOLD=1.54
    DEFAULT_MIN_Z_SCORE_THRESHOLD2=1.34
    KMEANS=1
elif [ "$SAMPLE_RATIO" == "0.125" ]; then
    DEFAULT_MIN_Z_SCORE_THRESHOLD=1.15
    DEFAULT_MIN_Z_SCORE_THRESHOLD2=1.54
    KMEANS=2
elif [ "$SAMPLE_RATIO" == "0.25" ]; then
    DEFAULT_MIN_Z_SCORE_THRESHOLD=0.67
    DEFAULT_MIN_Z_SCORE_THRESHOLD2=1.54
    KMEANS=4
fi
MIN_Z_SCORE_THRESHOLD="${MIN_Z_SCORE_THRESHOLD:-$DEFAULT_MIN_Z_SCORE_THRESHOLD}"
MIN_Z_SCORE_THRESHOLD2="${MIN_Z_SCORE_THRESHOLD2:-$DEFAULT_MIN_Z_SCORE_THRESHOLD2}"


if [ "$SCENARIO" == "15" ]; then
    EVAL_PERIOD=16000
    NUM_ITER=0.0625
    NUM_TASKS=8
elif [ "$SCENARIO" == "16" ]; then
    EVAL_PERIOD=16000
    NUM_ITER=0.125
    NUM_TASKS=4
elif [ "$SCENARIO" == "17" ]; then
    EVAL_PERIOD=16000
    NUM_ITER=0.0625
    NUM_TASKS=7
elif [ "$SCENARIO" == "12" ]; then
    EVAL_PERIOD=4000
    NUM_ITER=0.125
    NUM_TASKS=4
elif [ "$SCENARIO" == "18" ]; then
    EVAL_PERIOD=4000
    NUM_ITER=0.125
    NUM_TASKS=4
elif [ "$SCENARIO" == "20" ]; then
    EVAL_PERIOD=4000
    NUM_ITER=0.125
    NUM_TASKS=4
elif [ "$SCENARIO" == "8" ]; then
    EVAL_PERIOD=4000
    NUM_ITER=0.125
    NUM_TASKS=4
elif [ "$SCENARIO" == "102" ]; then
    EVAL_PERIOD=4000
    NUM_ITER=0.125
    NUM_TASKS=4
fi
# 0.1 -> 1.28, 0.2 -> 0.84, 0.3 -> 0.52

MAX_Z_SCORE_THRESHOLD=5
INFO_UPDATE_MODE="max"
EMA_UPDATE_MODE="none"
EMA_AVERAGE=False
EMA_POOL_SIZE=100


SOFTMAX_UPDATE_TEMP=0.5
# fed args
# NUM_ROUNDS=1
# NUM_CLIENTS=1
MODEL_MAX_LEN=20000
MEMORY_SIZE=10000000
IS_STREAMONLY=False
IS_SAR=False
IS_SAR_TASK=False

LORA_ENABLE=True
IA3_ENABLE=False

USE_TASK_ID=False
USE_PROMPT=False

# Do not write per-client optimizer/scheduler snapshots. train_VLM_CL.py saves
# the aggregated server model once at the end of each round.
SAVE_OPTIM=False

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

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="liuhaotian/llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-1b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-1B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-2b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-2B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-4b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-4B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-8b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-8B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-26b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-26B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3-1b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3-1B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3-2b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3-2B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3-8b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3-8B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3-9b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3-9B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
elif [ "$MODEL_ARCH" == "internvl3-14b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3-14B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    LR=4e-5
    BITS=16
else
    echo "Undefined setting"
    exit 1
fi

MM_PROJECTOR_LR=0 #3e-4
FINAL_LR=$LR #3e-4
MM_FINAL_LR=$LR #3e-4
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine" #cosine
WARMUP_RATIO=0.1 # SHOULD BE 0.03 / NUM_ROUNDS
DECAY_RATIO=0.9


if [ "$MODE" == "BudgetProb_v2" ]; then
    NOTE=MLLM_${MODEL_ARCH}_bs1_gradacc32_${MODE}_scenario${SCENARIO}_random${SAMPLE_RATIO}_seed${RND_SEED}_iter${NUM_ITER}_${SELECTION_METHOD}_GRADSAMPLE${GRAD_RATIO}_MINZSCORE${MIN_Z_SCORE_THRESHOLD}_EMA${EMA_RATIO}_STREAMONLY${IS_STREAMONLY}
elif [ "$MODE" == "oasis" ]; then
    NOTE=MLLM_${MODEL_ARCH}_bs1_gradacc32_${MODE}_scenario${SCENARIO}_random${SAMPLE_RATIO}_seed${RND_SEED}_iter${NUM_ITER}_${SELECTION_METHOD}_MINZSCORE${MIN_Z_SCORE_THRESHOLD}_MINZSCORE2${MIN_Z_SCORE_THRESHOLD2}_STREAMONLY${IS_STREAMONLY}
else
    NOTE=MLLM_${MODEL_ARCH}_bs1_gradacc32_${MODE}_scenario${SCENARIO}_random${SAMPLE_RATIO}_seed${RND_SEED}_iter${NUM_ITER}_${SELECTION_METHOD}_STREAMONLY${IS_STREAMONLY}
fi

# --master_port 29500
# --num_gpus=4
# LOAD_CHECKPOINT="client_states_fedours_bs4_saveoptim_lr6e-3_lastdownmean_freq5_fishercossimsoftmax_mean_sc0_4tasks_5rounds_fixitr100/server_model_round4.pth"
# LOAD_CHECKPOINT="client_states_fedours_bs4_saveoptim_lr6e-3_lastdownmean_freq5_fishercossimsoftmax_mean_sc0_4tasks_5rounds_fixitr100/server_model_round4.pth"
# # --deepspeed ./deepspeed_script/zero2.json \
# --num_rounds $NUM_ROUNDS \
# --num_tasks $NUM_TASKS \
run_training() {
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DEEPSPEED_BIN" --master_port $MASTER_PORT \
    --include localhost:$GPU \
    train_VLM_CL.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --model_max_length $MODEL_MAX_LEN \
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
    --grad_ratio $GRAD_RATIO \
    --output_dir "./results/test/"
}

# The experiment runner sets RUN_IN_BACKGROUND=0 and waits for this process.
# The default retains the original background-launch behavior.
if [ "${RUN_IN_BACKGROUND:-1}" = "1" ]; then
    run_training > ./nohup/${NOTE}.log 2>&1 &
else
    run_training
fi

# --selection_method $SELECTION_METHOD \

# --load_checkpoint $LOAD_CHECKPOINT \

# --eval_period $EVAL_PERIOD
# lr_scheduler_type
# --include localhost:7 \
