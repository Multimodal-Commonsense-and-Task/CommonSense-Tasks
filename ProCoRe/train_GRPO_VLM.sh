#!/bin/bash

export WANDB_DISABLED=true
export VLLM_LOGGING_LEVEL=DEBUG

usage() {
    cat <<'EOF'
Usage: bash train_GRPO_VLM.sh [--proposition-model MODEL]

Proposition timing examples:
  bash train_GRPO_VLM.sh --proposition-model gpt
  bash train_GRPO_VLM.sh --proposition-model precomputed

The "precomputed" model performs no model/API decomposition. It uses each
caption itself as its already-generated proposition.

For high-resolution timing, set STEP_TIMING_PATH. STOP_AFTER_STEPS can stop a
benchmark early without changing the full-run learning-rate schedule:
  STOP_AFTER_STEPS=50 STEP_TIMING_PATH=nohup/gpt_timing.jsonl \
    bash train_GRPO_VLM.sh --proposition-model gpt

Set PROPOSITION_PARALLELISM to control concurrent GPT requests or the local
Hugging Face decomposer batch size (default: 16):
  PROPOSITION_PARALLELISM=16 bash train_GRPO_VLM.sh

REWARD=diva selects DIVA-GRPO. The "diva" reward uses Rouge-L in place of the
official math-correctness reward. Its advantage recipe can be tuned without
editing this file:
  DIVA_WEIGHT_MODE=ZSCORENORM_WAN_RRBLOCAL_RRBGLOBAL \
  DIVA_WEIGHTED_ADVANTAGE_K=0.1 bash train_GRPO_VLM.sh

REWARD=pepo selects PEPO-G with the existing Rouge-L caption reward:
  REWARD=pepo PEPO_GATE_TEMPERATURE=1.8 bash train_GRPO_VLM.sh

REWARD=caprl selects the CapRL QA-utility reward: a text-only LLM answers
multiple-choice questions using nothing but the generated caption, and its
accuracy (x CAPRL_ACC_SCALE) is the reward.
  REWARD=caprl CAPRL_ANSWER_MODEL=Qwen/Qwen3-1.7B bash train_GRPO_VLM.sh

REWARD=caprlpp selects CapRL++, which adds two verifiable regularisers:
R = R_acc + CAPRL_FORMAT_WEIGHT * R_format + CAPRL_LENGTH_WEIGHT * R_len.
  REWARD=caprlpp CAPRL_FORMAT_MODE=pair CAPRL_LENGTH_L1=128 \
    bash train_GRPO_VLM.sh

REWARD=fast selects FAST-GRPO's difficulty-aware reward, adapted to paired
captions. Paired-caption Rouge-L is the task outcome, and FAST rewards concise
correct rollouts on easier groups and longer exploration on difficult groups:
  REWARD=fast FAST_CORRECTNESS_THRESHOLD=0.5 bash train_GRPO_VLM.sh

The question bank defaults to CAPRL_QA_SOURCE=contrastive, which derives the
MCQs from the gold caption pair with no extra model. CAPRL_QA_SOURCE=llm asks
CAPRL_QA_MODEL (gpt or a local HF model) to write CapRL-format questions and
caches them in CAPRL_QA_CACHE_PATH.
EOF
}

# CIL CONFIG
MODE="grpo"
# Overridable from the environment so the REWARD=<name> invocations documented
# in the usage text above actually select a reward (eval.sh already reads
# $REWARD the same way).
REWARD="${REWARD:-pepo}"
MODEL_ARCH="qwen3-8b" # "otter-9b" "idefics-8b" "internvl3_5-8b" "qwen3-8b" "onevision1_5-8b"
REWARD_WEIGHT=(1)
NUM_GEN=4

if [ "$REWARD" == "diva" ]; then
    REWARD_WEIGHT=(1)
elif [ "$REWARD" == "pepo" ]; then
    REWARD_WEIGHT=(1)
elif [ "$REWARD" == "vgpo" ]; then
    REWARD_WEIGHT=(1)
elif [ "$REWARD" == "rouge" ]; then
    REWARD_WEIGHT=(1)
elif [[ "$REWARD" == "caprl" || "$REWARD" == "caprlpp" ]]; then
    REWARD_WEIGHT=(1)
elif [ "$REWARD" == "fast" ]; then
    REWARD_WEIGHT=(1)
elif [ "$REWARD" == "vgv2" ]; then
    REWARD_WEIGHT=(1 4 2 4)
elif [ "$REWARD" == "procore" ]; then
    # Paper Eq. 6: lambda_I=2, lambda_T=1, lambda_s=4 (order: R_I, R_T, R_s)
    REWARD_WEIGHT=(2 1 4)
else
    echo "Undefined reward"
    exit 1
fi

# GRIT parameters (matching the official clipped-GRPO recipe).
GRIT_MAX_COMPLETION_LENGTH="${GRIT_MAX_COMPLETION_LENGTH:-1000}"
GRIT_CLIP_EPSILON="${GRIT_CLIP_EPSILON:-0.28}"

# DIVA-GRPO parameters ported from the official DIVA-GRPO recipe.
DIVA_VARIANT_NUM="${DIVA_VARIANT_NUM:-1}"
DIVA_WEIGHT_MODE="${DIVA_WEIGHT_MODE:-ZSCORENORM_WAN_RRBLOCAL_RRBGLOBAL}"
DIVA_SCORE_RANGES="${DIVA_SCORE_RANGES:-0.0:0.05,0.05:0.3,0.3:0.5,0.5:0.7,0.7:1.0}"
DIVA_DIFFICULTY_CHANGES="${DIVA_DIFFICULTY_CHANGES:-2,1,0,-1,-2}"
DIVA_MIN_DIFFICULTY="${DIVA_MIN_DIFFICULTY:-1}"
DIVA_MAX_DIFFICULTY="${DIVA_MAX_DIFFICULTY:-9}"
DIVA_INITIAL_DIFFICULTY="${DIVA_INITIAL_DIFFICULTY:-5}"
DIVA_WEIGHTED_ADVANTAGE_K="${DIVA_WEIGHTED_ADVANTAGE_K:-0.1}"
DIVA_RRB_THRESHOLD="${DIVA_RRB_THRESHOLD:-0.15}"
DIVA_RRB_SCALE_FACTOR="${DIVA_RRB_SCALE_FACTOR:-0.1}"
DIVA_LOCAL_ADVANTAGE_WEIGHT="${DIVA_LOCAL_ADVANTAGE_WEIGHT:-1.0}"
DIVA_GLOBAL_ADVANTAGE_WEIGHT="${DIVA_GLOBAL_ADVANTAGE_WEIGHT:-1.0}"

# PEPO-G parameters ported from the official PEPO recipe. PEPO is enabled
# only when REWARD=pepo; other reward modes retain their existing loss.
PEPO_GATE_ALPHA="${PEPO_GATE_ALPHA:-0.05}"
PEPO_GATE_TEMPERATURE="${PEPO_GATE_TEMPERATURE:-1.8}"
PEPO_USE_VISION_WEIGHTS="${PEPO_USE_VISION_WEIGHTS:-True}"
PEPO_VISION_SCORE_TYPE="${PEPO_VISION_SCORE_TYPE:-cosine}"
PEPO_IMAGE_TOKEN="${PEPO_IMAGE_TOKEN:-}"
PEPO_BETA="${PEPO_BETA:-0.001}"

# CapRL / CapRL++ parameters. The defaults mirror the official reward server
# (CapRL/CapRL_Training/reward_server/serve_rm.py and
# CapRL/CapRL++/train/verl/recipe/video_captionrl/), retargeted from dense
# image/video captions to this task's paired "caption_1 || caption_2" output.
CAPRL_QA_SOURCE="${CAPRL_QA_SOURCE:-contrastive}"   # contrastive | llm
CAPRL_QA_MODEL="${CAPRL_QA_MODEL:-gpt}"             # used when QA_SOURCE=llm
CAPRL_QA_CACHE_PATH="${CAPRL_QA_CACHE_PATH:-.cache/caprl/caprl_qa.sqlite3}"
CAPRL_QA_NUM="${CAPRL_QA_NUM:-4}"                   # CapRL --qa_num
CAPRL_QA_ALL="${CAPRL_QA_ALL:-False}"               # CapRL --all_qa
CAPRL_NUM_OPTIONS="${CAPRL_NUM_OPTIONS:-4}"
CAPRL_SHUFFLE_QA="${CAPRL_SHUFFLE_QA:-True}"        # CapRL --shuffle_qa
CAPRL_ANSWER_MODEL="${CAPRL_ANSWER_MODEL:-Qwen/Qwen3-1.7B}"
CAPRL_ANSWER_DEVICE="${CAPRL_ANSWER_DEVICE:-auto}"
CAPRL_ANSWER_BATCH_SIZE="${CAPRL_ANSWER_BATCH_SIZE:-32}"
CAPRL_ANSWER_TEMPERATURE="${CAPRL_ANSWER_TEMPERATURE:-0.6}"
CAPRL_ACC_SCALE="${CAPRL_ACC_SCALE:-2.0}"           # CapRL 1.0 scales acc by 2
CAPRL_FORMAT_MODE="${CAPRL_FORMAT_MODE:-pair}"      # pair | timestamp | none
CAPRL_FORMAT_WEIGHT="${CAPRL_FORMAT_WEIGHT:-0.2}"   # CapRL++ alpha
CAPRL_LENGTH_WEIGHT="${CAPRL_LENGTH_WEIGHT:-0.2}"   # CapRL++ beta
CAPRL_LENGTH_L1="${CAPRL_LENGTH_L1:-128}"           # CapRL++ REWARD_LENGTH_L1
CAPRL_LENGTH_L2="${CAPRL_LENGTH_L2:-256}"           # CapRL++ REWARD_LENGTH_L2
CAPRL_LENGTH_TOKENIZER="${CAPRL_LENGTH_TOKENIZER:-}"
CAPRL_DEBUG="${CAPRL_DEBUG:-False}"

# CapRL 1.0 has no format or length term; keep the run honest if the caller
# reuses a CapRL++ environment for a REWARD=caprl run.
if [ "$REWARD" == "caprl" ]; then
    CAPRL_FORMAT_WEIGHT=0
    CAPRL_LENGTH_WEIGHT=0
fi

# FAST-GRPO reward parameters. The official math-correctness signal is
# replaced by this project's paired-caption Rouge-L outcome; the remaining
# difficulty-aware format/length shaping follows FAST's reward recipe.
FAST_CORRECTNESS_THRESHOLD="${FAST_CORRECTNESS_THRESHOLD:-0.5}"
FAST_DIFFICULTY_PERCENTILE="${FAST_DIFFICULTY_PERCENTILE:-80.0}"
FAST_FORMAT_WEIGHT="${FAST_FORMAT_WEIGHT:-0.5}"
FAST_LENGTH_WEIGHT="${FAST_LENGTH_WEIGHT:-0.5}"
# Proposition decomposer choices: internlm/internlm2_5-1_8b,
# llama3_2-1b, llama3_2-3b, llama3_1-8b,
# qwen3-0_6b, qwen3-1_7b, qwen3-4b, qwen3-8b, flan-t5-base, flan-t5-large,
# flan-t5-xl, gpt, precomputed.
# This can be overridden without editing this file, for example:
#   PROPOSITION_MODEL=qwen3-0_6b bash train_GRPO_VLM.sh
# or with the command-line argument:
#   bash train_GRPO_VLM.sh --proposition-model precomputed
PROPOSITION_MODEL="${PROPOSITION_MODEL:-gpt}"
PROPOSITION_DEVICE="${PROPOSITION_DEVICE:-auto}"
PROPOSITION_CACHE_PATH="${PROPOSITION_CACHE_PATH:-.cache/vgv2/ground_truth_propositions.sqlite3}"
PROPOSITION_PARALLELISM="${PROPOSITION_PARALLELISM:-16}"
export PROPOSITION_PARALLELISM

if [[ ! "$PROPOSITION_PARALLELISM" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: PROPOSITION_PARALLELISM must be a positive integer." >&2
    exit 2
fi

while (($#)); do
    case "$1" in
        --proposition-model|--proposition_model)
            if (($# < 2)); then
                echo "Error: $1 requires a model name." >&2
                usage >&2
                exit 2
            fi
            PROPOSITION_MODEL="$2"
            shift 2
            ;;
        --proposition-model=*|--proposition_model=*)
            PROPOSITION_MODEL="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$DIVA_VARIANT_NUM" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: DIVA_VARIANT_NUM must be a positive integer." >&2
    exit 2
fi

# Set OPENAI_API_KEY (needed when PROPOSITION_MODEL=gpt) and HF_TOKEN in your shell; never hard-code them here.

DATASET_NAME="inpaintcoco" # openchair mhal-detect haloquest
EPOCH=1
GPU="${GPU:-4}"
MASTER_PORT="${MASTER_PORT:-24011}"
RUN_TAG="${RUN_TAG:-}"
MAX_STEPS="${MAX_STEPS:--1}"


# Don't Change Arguments Below-----------------------------------------------------------------
SCENARIO=3
RESUME_TRAIN_ROUND=-1
LOAD_CHECKPOINT="/disk1/mjlee/LLaVA/checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"

LORA_ENABLE=True
SHOT="original"
RND_SEED=1
USE_PROMPT=False

SAVE_OPTIM=True
NUM_ROUNDS=1
MODEL_MAX_LEN=20000
MEMORY_SIZE=10000000
IS_STREAMONLY=False


MUTUAL_PARAM=1
EVAL_PERIOD=100000
NUM_ITER=1
NUM_TASKS=1

GENERATOR_OUTPUT_SIZE=1024
GENERATOR_HIDDEN_DIM=8
GENERATOR_HIDDEN_FEATURE=8
KEY_EMBED_SIZE=64
POOL_SIZE=4
PROMPT_TOP_K=1

BATCHSIZE=1

if [ "$MODEL_ARCH" == "internvl3_5-8b" ]; then
    MODEL_NAME="OpenGVLab/InternVL3_5-8B"
    VERSION="v1"
    VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
    MODEL_TYPE="qwen"
    LR=5e-6
    BITS=16
elif [ "$MODEL_ARCH" == "otter-9b" ]; then
    MODEL_NAME="luodian/otter-9b-hf"
    VERSION="otter"
    VISION_TOWER="openai/clip-vit-large-patch14"
    MODEL_TYPE="otter"
    LR=2e-6
    BITS=16
    GRAD_CHECKPOINT=False
elif [ "$MODEL_ARCH" == "onevision1_5-8b" ]; then
    MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    VERSION="None"
    VISION_TOWER="None"
    MODEL_TYPE="None"
    LR=2e-5
    BITS=16
elif [ "$MODEL_ARCH" == "idefics-8b" ]; then
    MODEL_NAME="HuggingFaceM4/Idefics3-8B-Llama3"
    VERSION="llama"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    LR=5e-6
    MODEL_TYPE="llama3-8b"
    BITS=16
elif [ "$MODEL_ARCH" == "qwen3-8b" ]; then
    MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
    VERSION="None"
    VISION_TOWER="None"
    MODEL_TYPE="None"
    LR=2e-5
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
ATTEN_IMPLE="sdpa"

mkdir -p "nohup"
# NOTE=${MODEL_ARCH}_grpo_${DATASET_NAME}_${SHOT}_${REWARD}_RW${REWARD_WEIGHT[0]}_numgen${NUM_GEN}_scen${SCENARIO}_${LR}
if [[ "$REWARD" == "vgv2" || "$REWARD" == "procore" ]]; then
    PROPOSITION_TAG="${PROPOSITION_MODEL//\//_}"
    PROPOSITION_TAG="${PROPOSITION_TAG//:/_}"
    REWARD_TAG="${REWARD}_prop${PROPOSITION_TAG}"
else
    REWARD_TAG="$REWARD"
fi
if [[ "$REWARD" == "diva" ]]; then
    DIVA_TAG="_v${DIVA_VARIANT_NUM}-${DIVA_WEIGHT_MODE}-k${DIVA_WEIGHTED_ADVANTAGE_K}"
else
    DIVA_TAG=""
fi
if [[ "$REWARD" == "pepo" ]]; then
    PEPO_TAG="_${PEPO_VISION_SCORE_TYPE}-a${PEPO_GATE_ALPHA}-t${PEPO_GATE_TEMPERATURE}-b${PEPO_BETA}"
else
    PEPO_TAG=""
fi
if [[ "$REWARD" == "caprl" ]]; then
    CAPRL_ANSWER_TAG="${CAPRL_ANSWER_MODEL//\//_}"
    CAPRL_TAG="_${CAPRL_QA_SOURCE}-qa${CAPRL_QA_NUM}-${CAPRL_ANSWER_TAG}"
elif [[ "$REWARD" == "caprlpp" ]]; then
    CAPRL_ANSWER_TAG="${CAPRL_ANSWER_MODEL//\//_}"
    CAPRL_TAG="_${CAPRL_QA_SOURCE}-qa${CAPRL_QA_NUM}-${CAPRL_ANSWER_TAG}-${CAPRL_FORMAT_MODE}-a${CAPRL_FORMAT_WEIGHT}-b${CAPRL_LENGTH_WEIGHT}-l${CAPRL_LENGTH_L1}_${CAPRL_LENGTH_L2}"
else
    CAPRL_TAG=""
fi
if [[ "$REWARD" == "fast" ]]; then
    FAST_TAG="_t${FAST_CORRECTNESS_THRESHOLD}-p${FAST_DIFFICULTY_PERCENTILE}-f${FAST_FORMAT_WEIGHT}-l${FAST_LENGTH_WEIGHT}"
else
    FAST_TAG=""
fi
NOTE=${MODEL_ARCH}_grpo_${DATASET_NAME}_${SHOT}_${REWARD_TAG}_RW$(IFS=_; echo "${REWARD_WEIGHT[*]}")_numgen${NUM_GEN}_scen${SCENARIO}_${LR}${DIVA_TAG}${PEPO_TAG}${CAPRL_TAG}${FAST_TAG}
if [[ -n "$RUN_TAG" ]]; then
    RUN_TAG="${RUN_TAG//\//_}"
    RUN_TAG="${RUN_TAG//:/_}"
    RUN_TAG="${RUN_TAG// /_}"
    NOTE="${NOTE}_${RUN_TAG}"
fi

OUTPUT_DIR=model_states_${NOTE}
PEPO_IMAGE_TOKEN_ARGS=()
if [[ -n "$PEPO_IMAGE_TOKEN" ]]; then
    PEPO_IMAGE_TOKEN_ARGS=(--pepo_image_token "$PEPO_IMAGE_TOKEN")
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port $MASTER_PORT \
    --include localhost:$GPU \
    train_GRPO_VLM.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --dataset_name $DATASET_NAME \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --model_max_length $MODEL_MAX_LEN \
    --num_rounds $NUM_ROUNDS \
    --num_tasks $NUM_TASKS \
    --scenario $SCENARIO \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs $EPOCH \
    --max_steps "$MAX_STEPS" \
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
    --reward_weight "${REWARD_WEIGHT[@]}" \
    --is_streamonly $IS_STREAMONLY \
    --is_sar $IS_SAR \
    --is_sar_task $IS_SAR_TASK \
    --prompt_num 1 \
    --lora_enable $LORA_ENABLE \
    --use_task_id $USE_TASK_ID \
    --get_prompt $USE_PROMPT \
    --generator_output_size $GENERATOR_OUTPUT_SIZE \
    --generator_hidden_dim $GENERATOR_HIDDEN_DIM \
    --generator_hidden_feature $GENERATOR_HIDDEN_FEATURE \
    --key_embed_size $KEY_EMBED_SIZE \
    --pool_size $POOL_SIZE \
    --prompt_top_k $PROMPT_TOP_K \
    --save_optim $SAVE_OPTIM \
    --eval_period $EVAL_PERIOD \
    --resume_train_round $RESUME_TRAIN_ROUND \
    --load_checkpoint $LOAD_CHECKPOINT \
    --shot $SHOT \
    --reward_type $REWARD \
    --diva_variant_num "$DIVA_VARIANT_NUM" \
    --diva_weight_mode "$DIVA_WEIGHT_MODE" \
    --diva_score_ranges "$DIVA_SCORE_RANGES" \
    --diva_difficulty_changes "$DIVA_DIFFICULTY_CHANGES" \
    --diva_min_difficulty "$DIVA_MIN_DIFFICULTY" \
    --diva_max_difficulty "$DIVA_MAX_DIFFICULTY" \
    --diva_initial_difficulty "$DIVA_INITIAL_DIFFICULTY" \
    --diva_weighted_advantage_k "$DIVA_WEIGHTED_ADVANTAGE_K" \
    --diva_rrb_threshold "$DIVA_RRB_THRESHOLD" \
    --diva_rrb_scale_factor "$DIVA_RRB_SCALE_FACTOR" \
    --diva_local_advantage_weight "$DIVA_LOCAL_ADVANTAGE_WEIGHT" \
    --diva_global_advantage_weight "$DIVA_GLOBAL_ADVANTAGE_WEIGHT" \
    --pepo_gate_alpha "$PEPO_GATE_ALPHA" \
    --pepo_gate_temperature "$PEPO_GATE_TEMPERATURE" \
    --pepo_use_vision_weights "$PEPO_USE_VISION_WEIGHTS" \
    --pepo_vision_score_type "$PEPO_VISION_SCORE_TYPE" \
    "${PEPO_IMAGE_TOKEN_ARGS[@]}" \
    --pepo_beta "$PEPO_BETA" \
    --caprl_qa_source "$CAPRL_QA_SOURCE" \
    --caprl_qa_model "$CAPRL_QA_MODEL" \
    --caprl_qa_cache_path "$CAPRL_QA_CACHE_PATH" \
    --caprl_qa_num "$CAPRL_QA_NUM" \
    --caprl_qa_all "$CAPRL_QA_ALL" \
    --caprl_num_options "$CAPRL_NUM_OPTIONS" \
    --caprl_shuffle_qa "$CAPRL_SHUFFLE_QA" \
    --caprl_answer_model "$CAPRL_ANSWER_MODEL" \
    --caprl_answer_device "$CAPRL_ANSWER_DEVICE" \
    --caprl_answer_batch_size "$CAPRL_ANSWER_BATCH_SIZE" \
    --caprl_answer_temperature "$CAPRL_ANSWER_TEMPERATURE" \
    --caprl_acc_scale "$CAPRL_ACC_SCALE" \
    --caprl_format_mode "$CAPRL_FORMAT_MODE" \
    --caprl_format_weight "$CAPRL_FORMAT_WEIGHT" \
    --caprl_length_weight "$CAPRL_LENGTH_WEIGHT" \
    --caprl_length_l1 "$CAPRL_LENGTH_L1" \
    --caprl_length_l2 "$CAPRL_LENGTH_L2" \
    --caprl_length_tokenizer "$CAPRL_LENGTH_TOKENIZER" \
    --caprl_debug "$CAPRL_DEBUG" \
    --fast_correctness_threshold "$FAST_CORRECTNESS_THRESHOLD" \
    --fast_difficulty_percentile "$FAST_DIFFICULTY_PERCENTILE" \
    --fast_format_weight "$FAST_FORMAT_WEIGHT" \
    --fast_length_weight "$FAST_LENGTH_WEIGHT" \
    --proposition_model "$PROPOSITION_MODEL" \
    --proposition_device "$PROPOSITION_DEVICE" \
    --proposition_cache_path "$PROPOSITION_CACHE_PATH" \
    --attn_impl $ATTEN_IMPLE \
    --grpo_num_gen $NUM_GEN \
    --output_dir "./results/test/" > ./nohup/${NOTE}.log 2>&1 &

TRAIN_PID=$!
if [[ "${WAIT_FOR_TRAINING:-false}" == "true" ]]; then
    wait "$TRAIN_PID"
fi
