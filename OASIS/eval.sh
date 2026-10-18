
#!/bin/bash
# CIL CONFIG
MODE="sft_org"
MODEL_ARCH="internvl2_5-1b" # llava bunny_3b bunny_8b

# fed args
RND_SEED=1
SELECTION_RATIO=0.0625
SCENARIO=20
IS_STREAMONLY=False
NUM_ITER=0.125


SELECTION_METHOD="fisher"
MIN_Z_SCORE_THRESHOLD=1.54
EMA_RATIO=0.9


NUM_ROUNDS=1
NUM_CLIENTS=1
MODEL_MAX_LEN=20000
MAX_NEW_TOKENS=512
SETUP="disjoint"


NOTE=MLLM_${MODEL_ARCH}_bs1_gradacc32_${MODE}_scenario${SCENARIO}_random${SELECTION_RATIO}_seed${RND_SEED}_iter${NUM_ITER}_${SELECTION_METHOD}_MINZSCORE${MIN_Z_SCORE_THRESHOLD}_EMA${EMA_RATIO}_STREAMONLY${IS_STREAMONLY}


if [ "$SCENARIO" == "16" ]; then
    NUM_TASKS=4
elif [ "$SCENARIO" == "17" ]; then
    NUM_TASKS=7
elif [ "$SCENARIO" == "20" ]; then
    NUM_TASKS=4
fi

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="liuhaotian/llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16
elif [ "$MODEL_ARCH" == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-1b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-1B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-2b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-2B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
    BITS=16
elif [ "$MODEL_ARCH" == "internvl2_5-4b" ]; then
    MODEL_NAME="OpenGVLab/InternVL2_5-4B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="llama"
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
    BITS=16
else
    echo "Undefined setting"
    exit 1
fi

# ROUND_TO_EVALS=$2
ROUND_TO_EVALS=(1 2 3 4)
ITER_TO_EVAL=2400

for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$1 python eval_VLM_CL.py \
        --is_eval True \
        --model_name_or_path $MODEL_NAME \
        --model_name_for_dataarg $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --version $VERSION \
        --scenario $SCENARIO \
        --num_rounds $NUM_ROUNDS \
        --num_tasks $NUM_TASKS \
        --num_clients $NUM_CLIENTS \
        --model_max_length $MODEL_MAX_LEN \
        --max_new_tokens $MAX_NEW_TOKENS \
        --vision_tower $VISION_TOWER \
        --bits $BITS \
        --bf16 True \
        --tf32 True \
        --note $NOTE \
        --mode $MODE \
        --eval_server True \
        --unseen_task True \
        --zeroshot False \
        --lora_enable True \
        --ia3_enable False \
        --generator_output_size 512 \
        --generator_hidden_dim 8 \
        --generator_hidden_feature 8 \
        --key_embed_size 64 \
        --prompt_top_k 1 \
        --pool_size 40 \
        --set_state "gate" \
        --is_prompt False \
        --use_task_vector False \
        --round_to_eval ${ROUND_TO_EVALS[$index]} \
        --eval_iter $ITER_TO_EVAL \
        --setup $SETUP \
        --output_dir "./nohup" #> ./nohup/eval_${NOTE}_eval_round${ROUND_TO_EVALS[$index]}_iter${ITER_TO_EVAL}.log 2>&1 & #_iter${ITER_TO_EVAL
done
# --eval_period $EVAL_PERIOD