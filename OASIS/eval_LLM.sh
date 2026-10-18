# #!/bin/bash
# #SBATCH --time=12:00:00
# #SBATCH --job-name=icl_eval
# #SBATCH --mem=32G
# #SBATCH -p gpu
# #SBATCH --ntasks=1
# #SBATCH --nodes 1
# #SBATCH --gres=gpu:1
# #SBATCH --account=b2025b08-038-users

# module load Anaconda3
# # source activate /ceph/hpc/data/d2025d07-105-users/icl
# source activate /ceph/hpc/data/d2025d07-105-users/qwen
# export HF_HOME=/ceph/hpc/data/d2025d07-105-users/LMM-ICL/model_ckpt

#!/bin/bash
# CIL CONFI
# NOTE="LMM_qwen_bs1_gradacc32_sft_scenario1_10000_random0_125_seed1"
MODE="sft"
MODEL_ARCH="llama-8b" # llava bunny_3b bunny_8b
RND_SEED=1
SELECTION_RATIO=0.0625
ONLINE_ITER=0.25

SELECTION_METHOD="fisher"

LR=2e-7
ROUND_TO_EVALS=(1)
# fed args
SCENARIO=1
NUM_ROUNDS=1
NUM_CLIENTS=1
MODEL_MAX_LEN=20000
MAX_NEW_TOKENS=512
SETUP="disjoint"

# NOTE=LLM_${MODEL_ARCH}_bs1_gradacc32_${MODE}_scenario${SCENARIO}_random${SELECTION_RATIO}_seed${RND_SEED}_iter${ONLINE_ITER}_selfsup

if [ "$SCENARIO" == "1" ]; then
    NUM_TASKS=15
elif [ "$SCENARIO" == "2" ]; then
    NUM_TASKS=15
elif [ "$SCENARIO" == "3" ]; then
    NUM_TASKS=8
elif [ "$SCENARIO" == "4" ]; then
    NUM_TASKS=19
fi

if [ "$MODEL_ARCH" == "llama-8b" ]; then
    MODEL_NAME="meta-llama/Llama-3.1-8B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    # LR=1e-5 
    # LR=2e-4
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "llama-1b" ]; then
    MODEL_NAME="meta-llama/Llama-3.2-1B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "llama-3b" ]; then
    MODEL_NAME="meta-llama/Llama-3.2-3B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "qwen-8b" ]; then
    MODEL_NAME="Qwen/Qwen3-8B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "qwen-0_6b" ]; then
    MODEL_NAME="Qwen/Qwen3-0_6B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16
    IS_MULTIMODAL=False
elif [ "$MODEL_ARCH" == "qwen-4b" ]; then
    MODEL_NAME="Qwen/Qwen3-4B"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16
    IS_MULTIMODAL=False
else
    echo "Undefined setting"
    exit 1
fi

NOTE=LLM_${MODEL_ARCH}_${LR}_bs1_gradacc32_${MODE}_scenario${SCENARIO}_random${SELECTION_RATIO}_seed${RND_SEED}_iter${ONLINE_ITER}_${SELECTION_METHOD}

# ROUND_TO_EVALS=$2
ITER_TO_EVAL=2400

for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$1 python eval_LLM_CL.py \
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
        --is_multimodal $IS_MULTIMODAL \
        --output_dir "./nohup"  # > ./nohup/eval_${NOTE}_eval_round${ROUND_TO_EVALS[$index]}_iter${ITER_TO_EVAL}.log 2>&1 & #_iter${ITER_TO_EVAL
done
# --eval_period $EVAL_PERIOD