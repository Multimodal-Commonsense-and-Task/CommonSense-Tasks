#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
# CIL CONFIG
NOTE="sft"
MODE="sft"

# fed args
SCENARIO=Fed-Scope_hetero_llama_3B_8B
NUM_ROUNDS=5
NUM_TASKS=4
NUM_CLIENTS=5
MODEL_MAX_LEN=20000
MAX_NEW_TOKENS=512

MODEL_NAME="thkim0305/llama3.2_1B_vl"
VERSION="llama3"
MODEL_TYPE="llama3"
BITS=16

# ROUND_TO_EVALS=$2
ROUND_TO_EVALS=(20)
ITER_TO_EVAL=0

for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$1 python -m eval_scripts.eval_FS_LLM_math \
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
        --bits $BITS \
        --bf16 True \
        --tf32 True \
        --note $NOTE \
        --mode $MODE \
        --unseen_task False \
        --zeroshot False \
        --lora_enable True \
        --set_state "gate" \
        --use_task_vector False \
        --is_multimodal False \
        --lora_r 16 \
        --lora_alpha 32 \
        --round_to_eval ${ROUND_TO_EVALS[$index]} \
        --output_dir "./nohup" > ./nohup/${NOTE}_eval_round${ROUND_TO_EVALS[$index]}_math.log 2>&1 & #_iter${ITER_TO_EVAL}
done
# --eval_period $EVAL_PERIOD
#--eval_iter $ITER_TO_EVAL \