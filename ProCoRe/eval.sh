#!/usr/bin/env bash

# =============================================================================
# Unified eval launcher. Combines eval_VLM.sh (base), eval_VLM_volcano.sh,
# eval_VLM_vicl.sh and eval_VLM_sparc.sh into one script.
#
# GRPO examples:
#   bash eval.sh diva                 # DIVA on GPU 0
#   bash eval.sh pepo 1               # PEPO on GPU 1
#   bash eval.sh both 0               # DIVA, then PEPO, on GPU 0
#   bash eval.sh 1 pepo               # GPU-first form is also accepted
#
# EVAL_REWARD and GPU provide equivalent environment-variable overrides.
# GRPO_NOTE can select an exact training NOTE when a run used custom settings.
#
# Edit the experiment knobs (MODEL_ARCHS, EVAL_SHOTS, DATASET_NAME, ORDERS,
# METHOD, ...) directly below, as in the original scripts.
#
# Each variant only changes: the python entrypoint, the model-name suffix, and
# a couple of variant-specific flags (see the case block below).
# =============================================================================

usage() {
    cat <<'EOF'
Usage:
  bash eval.sh diva [GPU]
  bash eval.sh pepo [GPU]
  bash eval.sh caprl [GPU]
  bash eval.sh caprlpp [GPU]
  bash eval.sh both [GPU]
  bash eval.sh [GPU] diva|pepo|caprl|caprlpp

Optional overrides:
  VARIANT=base|volcano|vicl|sparc
  EVAL_REWARD=diva|pepo|caprl|caprlpp|rouge|vgpo|vgv2|procore
  GRPO_NOTE=<exact training NOTE>
  RUN_TAG=<training RUN_TAG>

CapRL caption metrics (QA utility, format, length) can be reported for any
checkpoint, not just CapRL-trained ones:
  CAPRL_EVAL=True bash eval.sh pepo 7
They are always reported for caprl/caprlpp runs. The CapRL_* variables below
control the question bank and the text-only answerer.
EOF
}

VARIANT="${VARIANT:-base}"
GPU="${GPU:-0}"
GRPO_REWARD="${EVAL_REWARD:-${REWARD:-diva}}"

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    diva|pepo|both|rouge|vgpo|vgv2|procore|caprl|caprlpp)
        GRPO_REWARD="$1"
        GPU="${2:-$GPU}"
        ;;
    "")
        ;;
    *)
        GPU="$1"
        GRPO_REWARD="${2:-$GRPO_REWARD}"
        ;;
esac

# Optional conda env (used for SLURM batch runs, as in the old volcano script).
# Leave empty to skip activation (e.g. when running in an already-active env).
# --- per-variant settings -----------------------------------------------------
EXTRA_ARGS=()
case "$VARIANT" in
    base)
        EVAL_SCRIPT="eval_VLM.py";          MODEL_SUFFIX="" ;;
    volcano)
        EVAL_SCRIPT="eval_VLM_volcano.py";  MODEL_SUFFIX="_volcano" ;;
    vicl)
        EVAL_SCRIPT="eval_VLM_vicl.py";     MODEL_SUFFIX="_vicl" ;;
    sparc)
        EVAL_SCRIPT="eval_VLM_sparc.py";    MODEL_SUFFIX="_sparc"
        EXTRA_ARGS=(--use_sparc True \
            --sparc_alpha 1.1 --sparc_beta 0.1 --sparc_tau 1.5 \
            --sparc_selected_layer 20 --sparc_start_layer 0 --sparc_end_layer 31) ;;
    *)
        echo "Unknown variant: '$VARIANT' (expected base|volcano|vicl|sparc)"; exit 1 ;;
esac

MODEL_ARCHS=("qwen3-8b") # "qwen-7b" "onevision-7b" "internvl2_5-8b" "otter-9b" "idefics2-8b" "idefics-8b" "internvl3_5-8b" "qwen3-8b" "onevision1_5-8b"
# EVAL_SHOTS=("0-shot" "1-shot" "2-shot" "8-shot" "4-shot") #"8-shot" "2-shot" "16-shot" "4-shot"
DATASET_NAME="${DATASET_NAME:-storystreamv2}" #mic vliclv2 truemicl isekai stanford10000 stanford10000_5 stanford10000_+1 pororo flintstones story_visual_ICL storystream movie101 moviegraphs memor illusionvqa-comp
ORDERS=("random") # "mmices_asc" "multimodal" "random"
METHOD="None" # Finetune None dara mimic fv mtv grpo mt-icl

if [[ "$METHOD" == "grpo" && "$GRPO_REWARD" == "both" ]]; then
    bash "$0" diva "$GPU" || exit $?
    bash "$0" pepo "$GPU"
    exit $?
fi


if [ "$DATASET_NAME" = "mic" ]; then
    SCENARIOS=(1 2 3 4 5 6 7 8 9 10) # 1 2 3 4 5 6 7 8 9 10
    EVAL_SHOTS=("0-shot" "1-shot" "2-shot" "4-shot" "8-shot") 
elif [ "$DATASET_NAME" = "vliclv2" ]; then
    SCENARIOS=(1 2 3 4) # 1 2 3 4
    EVAL_SHOTS=("0-shot" "1-shot" "2-shot" "4-shot" "8-shot")  
elif [ "$DATASET_NAME" = "truemicl" ]; then
    SCENARIOS=(1 2 3 4 5 6 8) # 1 2 3 4 5 6 8
    EVAL_SHOTS=("0-shot" "1-shot" "2-shot" "4-shot" "8-shot")  
elif [ "$DATASET_NAME" = "flintstonesv3" ] || [ "$DATASET_NAME" = "storystreamv2" ] || [ "$DATASET_NAME" = "pororov4" ]; then
    SCENARIOS=(1) # 1
    EVAL_SHOTS=("0-shot" "1-shot" "2-shot" "3-shot" "4-shot") 
elif [ "$DATASET_NAME" = "flintstonesv3_new" ] || [ "$DATASET_NAME" = "storystreamv2_new" ] || [ "$DATASET_NAME" = "pororov4_new" ]; then
    SCENARIOS=(1) # 1
    EVAL_SHOTS=("2-shot" "3-shot" "4-shot") 
elif [ "$DATASET_NAME" = "isekai" ]; then
    SCENARIOS=(1) # 1
    EVAL_SHOTS=("0-shot" "2-shot" "4-shot" "8-shot" "12-shot")   
elif [ "$DATASET_NAME" = "story_visual_ICLv2" ]; then
    SCENARIOS=(1)
    EVAL_SHOTS=("0-shot" "1-shot" "2-shot" "3-shot" "4-shot")
elif [ "$DATASET_NAME" = "illusionvqa-comp" ] || [ "$DATASET_NAME" = "idk" ] || [ "$DATASET_NAME" = "idk_mix" ] || [ "$DATASET_NAME" = "refcoco" ] || [ "$DATASET_NAME" = "inpaintcoco" ] || [ "$DATASET_NAME" = "AMBER" ] || [ "$DATASET_NAME" = "openchair" ] || [ "$DATASET_NAME" = "clear" ] || [ "$DATASET_NAME" = "kie" ]; then
    SCENARIOS=(1) # 1
elif [ "$DATASET_NAME" = "memor" ] || [ "$DATASET_NAME" = "moviegraphs" ]; then
    SCENARIOS=(1 2) # 1 2
elif [ "$DATASET_NAME" = "SMMILE" ]; then
    SCENARIOS=(2) # 2
elif [ "$DATASET_NAME" = "pope" ]; then
    SCENARIOS=(1 2 3) # 1 2 3
elif [ "$DATASET_NAME" = "pororo_textinertia" ] || [ "$DATASET_NAME" = "pororo_textinertia2" ] || [ "$DATASET_NAME" = "pororo_textinertia3" ] || [ "$DATASET_NAME" = "pororo_textinertia4" ]; then
    SCENARIOS=(3) # 1 2 3
fi


# Don't Change Arguments Below-----------------------------------------------------------------
EPOCH=1
TRAIN_SHOT="mixed"
REWARD_WEIGHT=(2 2 2 2)
TRAIN_SCENARIO=0
TRAIN_DATASET_NAME=$DATASET_NAME # openchair mhal-detect haloquest inpaintcoco winoground
# TRAIN_DATASET_NAME=truemicl

if [ "$METHOD" == "mimic" ] || [ "$METHOD" == "mimic_static" ] || [ "$METHOD" == "mimic_mixed" ]; then
    LR=5e-3
elif [ "$METHOD" == "dara" ]; then
    LR=1e-3
elif [ "$METHOD" == "grpo" ]; then
    # ==== These MUST match train_GRPO_VLM.sh so eval loads the saved model ====
    # Saved at ./client_states_<NOTE>/server_model_round0.pth, where <NOTE> is
    # rebuilt below (search "grpo" in the NOTE section). Mirror every knob here.
    LR=2e-5                            # train_GRPO_VLM.sh LR for this MODEL_ARCH
    REWARD="$GRPO_REWARD"               # train_GRPO_VLM.sh REWARD
    case "$REWARD" in
        diva|pepo|vgpo|rouge|caprl|caprlpp)
            REWARD_WEIGHT=(1)
            ;;
        vgv2)
            REWARD_WEIGHT=(1 4 2 4)
            ;;
        procore)
            REWARD_WEIGHT=(2 1 4)
            ;;
        *)
            echo "Unknown GRPO reward: '$REWARD'" >&2
            usage >&2
            exit 2
            ;;
    esac
    NUM_GEN=4                          # train_GRPO_VLM.sh NUM_GEN
    TRAIN_SCENARIO=3                   # train_GRPO_VLM.sh SCENARIO
    TRAIN_SHOT="original"             # train_GRPO_VLM.sh SHOT
    PROPOSITION_MODEL="${PROPOSITION_MODEL:-gpt}"
    TRAIN_DATASET_NAME="inpaintcoco"  # train_GRPO_VLM.sh DATASET_NAME

    # These values affect the training NOTE and therefore the checkpoint path.
    DIVA_VARIANT_NUM="${DIVA_VARIANT_NUM:-1}"
    DIVA_WEIGHT_MODE="${DIVA_WEIGHT_MODE:-ZSCORENORM_WAN_RRBLOCAL_RRBGLOBAL}"
    DIVA_WEIGHTED_ADVANTAGE_K="${DIVA_WEIGHTED_ADVANTAGE_K:-0.1}"
    PEPO_GATE_ALPHA="${PEPO_GATE_ALPHA:-0.05}"
    PEPO_GATE_TEMPERATURE="${PEPO_GATE_TEMPERATURE:-1.8}"
    PEPO_VISION_SCORE_TYPE="${PEPO_VISION_SCORE_TYPE:-cosine}"
    PEPO_BETA="${PEPO_BETA:-0.001}"
    # CapRL / CapRL++ knobs. These MUST match train_GRPO_VLM.sh: they are part
    # of the training NOTE and therefore of the checkpoint path.
    CAPRL_QA_SOURCE="${CAPRL_QA_SOURCE:-contrastive}"
    CAPRL_QA_NUM="${CAPRL_QA_NUM:-4}"
    CAPRL_ANSWER_MODEL="${CAPRL_ANSWER_MODEL:-Qwen/Qwen3-1.7B}"
    CAPRL_FORMAT_MODE="${CAPRL_FORMAT_MODE:-pair}"
    CAPRL_FORMAT_WEIGHT="${CAPRL_FORMAT_WEIGHT:-0.2}"
    CAPRL_LENGTH_WEIGHT="${CAPRL_LENGTH_WEIGHT:-0.2}"
    CAPRL_LENGTH_L1="${CAPRL_LENGTH_L1:-128}"
    CAPRL_LENGTH_L2="${CAPRL_LENGTH_L2:-256}"
    if [ "$REWARD" == "caprl" ]; then
        CAPRL_FORMAT_WEIGHT=0
        CAPRL_LENGTH_WEIGHT=0
    fi
    RUN_TAG="${RUN_TAG:-}"
    GRPO_NOTE="${GRPO_NOTE:-}"
elif [ "$METHOD" == "Finetune" ]; then
    TRAIN_SCENARIO=2
    TRAIN_DATASET_NAME="inpaintcoco" # openchair mhal-detect haloquest inpaintcoco winoground
    TRAIN_SHOT="original"
elif [ "$METHOD" == "lrv" ]; then
    TRAIN_SCENARIO=2
    TRAIN_DATASET_NAME="lrv" # openchair mhal-detect haloquest inpaintcoco winoground
    TRAIN_SHOT="original"
fi

# --- CapRL caption metrics (eval side) ---------------------------------------
# Reported automatically for caprl/caprlpp checkpoints, and for any other run
# when CAPRL_EVAL=True. These do not affect the checkpoint path.
if [[ "$METHOD" == "grpo" && ( "$REWARD" == "caprl" || "$REWARD" == "caprlpp" ) ]]; then
    CAPRL_EVAL="${CAPRL_EVAL:-True}"
else
    CAPRL_EVAL="${CAPRL_EVAL:-False}"
fi
CAPRL_QA_SOURCE="${CAPRL_QA_SOURCE:-contrastive}"
CAPRL_QA_MODEL="${CAPRL_QA_MODEL:-gpt}"
CAPRL_QA_CACHE_PATH="${CAPRL_QA_CACHE_PATH:-.cache/caprl/caprl_qa.sqlite3}"
CAPRL_QA_NUM="${CAPRL_QA_NUM:-4}"
CAPRL_QA_ALL="${CAPRL_QA_ALL:-False}"
CAPRL_NUM_OPTIONS="${CAPRL_NUM_OPTIONS:-4}"
CAPRL_SHUFFLE_QA="${CAPRL_SHUFFLE_QA:-True}"
CAPRL_ANSWER_MODEL="${CAPRL_ANSWER_MODEL:-Qwen/Qwen3-1.7B}"
CAPRL_ANSWER_DEVICE="${CAPRL_ANSWER_DEVICE:-auto}"
CAPRL_ANSWER_BATCH_SIZE="${CAPRL_ANSWER_BATCH_SIZE:-32}"
# Eval is a measurement, not a training signal: answer greedily by default so
# the reported number is reproducible.
CAPRL_ANSWER_TEMPERATURE="${CAPRL_ANSWER_TEMPERATURE:-0.0}"
CAPRL_ACC_SCALE="${CAPRL_ACC_SCALE:-2.0}"
CAPRL_FORMAT_MODE="${CAPRL_FORMAT_MODE:-pair}"
CAPRL_FORMAT_WEIGHT="${CAPRL_FORMAT_WEIGHT:-0.2}"
CAPRL_LENGTH_WEIGHT="${CAPRL_LENGTH_WEIGHT:-0.2}"
CAPRL_LENGTH_L1="${CAPRL_LENGTH_L1:-128}"
CAPRL_LENGTH_L2="${CAPRL_LENGTH_L2:-256}"
CAPRL_LENGTH_TOKENIZER="${CAPRL_LENGTH_TOKENIZER:-}"
CAPRL_DEBUG="${CAPRL_DEBUG:-False}"

RW_STR=$(IFS=_; echo "${REWARD_WEIGHT[*]}")
# Same sanitization as train_GRPO_VLM.sh so the grpo NOTE matches the saved dir.
PROPOSITION_TAG="${PROPOSITION_MODEL//\//_}"
PROPOSITION_TAG="${PROPOSITION_TAG//:/_}"

if [[ "$METHOD" == "grpo" ]]; then
    if [[ "$REWARD" == "vgv2" || "$REWARD" == "procore" ]]; then
        REWARD_TAG="${REWARD}_prop${PROPOSITION_TAG}"
    else
        REWARD_TAG="$REWARD"
    fi

    if [[ "$REWARD" == "diva" ]]; then
        GRPO_METHOD_TAG="_v${DIVA_VARIANT_NUM}-${DIVA_WEIGHT_MODE}-k${DIVA_WEIGHTED_ADVANTAGE_K}"
    elif [[ "$REWARD" == "pepo" ]]; then
        GRPO_METHOD_TAG="_${PEPO_VISION_SCORE_TYPE}-a${PEPO_GATE_ALPHA}-t${PEPO_GATE_TEMPERATURE}-b${PEPO_BETA}"
    elif [[ "$REWARD" == "caprl" ]]; then
        CAPRL_ANSWER_TAG="${CAPRL_ANSWER_MODEL//\//_}"
        GRPO_METHOD_TAG="_${CAPRL_QA_SOURCE}-qa${CAPRL_QA_NUM}-${CAPRL_ANSWER_TAG}"
    elif [[ "$REWARD" == "caprlpp" ]]; then
        CAPRL_ANSWER_TAG="${CAPRL_ANSWER_MODEL//\//_}"
        GRPO_METHOD_TAG="_${CAPRL_QA_SOURCE}-qa${CAPRL_QA_NUM}-${CAPRL_ANSWER_TAG}-${CAPRL_FORMAT_MODE}-a${CAPRL_FORMAT_WEIGHT}-b${CAPRL_LENGTH_WEIGHT}-l${CAPRL_LENGTH_L1}_${CAPRL_LENGTH_L2}"
    else
        GRPO_METHOD_TAG=""
    fi

    SANITIZED_RUN_TAG="${RUN_TAG//\//_}"
    SANITIZED_RUN_TAG="${SANITIZED_RUN_TAG//:/_}"
    SANITIZED_RUN_TAG="${SANITIZED_RUN_TAG// /_}"

    build_grpo_note() {
        local model_arch="$1"
        local note
        note="${model_arch}_grpo_${TRAIN_DATASET_NAME}_${TRAIN_SHOT}_${REWARD_TAG}_RW${RW_STR}_numgen${NUM_GEN}_scen${TRAIN_SCENARIO}_${LR}${GRPO_METHOD_TAG}"
        if [[ -n "$SANITIZED_RUN_TAG" ]]; then
            note="${note}_${SANITIZED_RUN_TAG}"
        fi
        printf '%s' "$note"
    }
fi

MODE="sft"
USE_VIS_TOKEN_AMPLIFIER=False
USE_PAI=False
if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
    VIS_TOKEN_AMPLIFIERS=(0.1 0.3 0.5 0.7 1.0 1.5 2.0)
elif [ "$USE_PAI" == "True" ]; then
    VIS_TOKEN_AMPLIFIERS=(0.1)
else
    VIS_TOKEN_AMPLIFIERS=(0.0)
fi

USE_ICOT=False
USE_COTPROMPT=False
COTPROMPT_VER=2
if [ "$USE_ICOT" == "True" ]; then
    USE_COTPROMPT=True
fi

NUM_ROUNDS=1
NUM_TASKS=1
MODEL_MAX_LEN=40000
MAX_NEW_TOKENS=512

ROUND_TO_EVALS=(1)
ITER_TO_EVAL=0
# eval_onevision.py
# eval_VLM_CL.py
# CUDA_VISIBLE_DEVICES=$1
for ((ind5=0; ind5<${#VIS_TOKEN_AMPLIFIERS[@]}; ind5++)); do
    VIS_TOKEN_AMPLIFIER=${VIS_TOKEN_AMPLIFIERS[$ind5]}
    for ((ind3=0; ind3<${#EVAL_SHOTS[@]}; ind3++)); do
        EVAL_SHOT=${EVAL_SHOTS[$ind3]}
        for ((ind4=0; ind4<${#MODEL_ARCHS[@]}; ind4++)); do
            MODEL_ARCH=${MODEL_ARCHS[$ind4]}
            if [ "$MODEL_ARCH" == "llava" ]; then
                MODEL_NAME="liuhaotian/llava-v1.5-7b"
                MODEL_NAME_OFF="llava-7b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "llava13b" ]; then
                MODEL_NAME="liuhaotian/llava-v1.5-13b"
                MODEL_NAME_OFF="llava-13b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-8b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-8B"
                MODEL_NAME_OFF="internvl2_5-8b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-4b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-4B"
                MODEL_NAME_OFF="internvl2_5-4b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-2b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-2B"
                MODEL_NAME_OFF="internvl2_5-2b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl2_5-1b" ]; then
                MODEL_NAME="OpenGVLab/InternVL2_5-1B"
                MODEL_NAME_OFF="internvl2_5-1b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-8b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-8B"
                MODEL_NAME_OFF="internvl3-8b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-1b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-1B"
                MODEL_NAME_OFF="internvl3-1b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-2b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-2B"
                MODEL_NAME_OFF="internvl3-2b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-9b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-9B"
                MODEL_NAME_OFF="internvl3-9b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3-14b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3-14B"
                MODEL_NAME_OFF="internvl3-14b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-8b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-8B"
                MODEL_NAME_OFF="internvl3_5-8b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-14b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-14B"
                MODEL_NAME_OFF="internvl3_5-14b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-4b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-4B"
                MODEL_NAME_OFF="internvl3_5-4b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-2b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-2B"
                MODEL_NAME_OFF="internvl3_5-2b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "internvl3_5-1b" ]; then
                MODEL_NAME="OpenGVLab/InternVL3_5-1B"
                MODEL_NAME_OFF="internvl3_5-1b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="OpenGVLab/InternViT-300M-448px-V2_5"
                MODEL_TYPE="qwen"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen-3b" ]; then
                MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
                MODEL_NAME_OFF="qwen-3b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen-7b" ]; then
                MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
                MODEL_NAME_OFF="qwen-7b${MODEL_SUFFIX}"
                VERSION="v1"
                VISION_TOWER="openai/clip-vit-large-patch14-336"
                MODEL_TYPE="llama"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision-0_5b" ]; then
                MODEL_NAME="lmms-lab/llava-onevision-qwen2-0.5b-ov"
                MODEL_NAME_OFF="onevision-0_5b${MODEL_SUFFIX}"
                VERSION="llama"
                VISION_TOWER="google/siglip-so400m-patch14-384"
                MODEL_TYPE="llama3-8b"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision-7b" ]; then
                MODEL_NAME="lmms-lab/llava-onevision-qwen2-7b-ov"
                MODEL_NAME_OFF="onevision-7b${MODEL_SUFFIX}"
                VERSION="llama"
                VISION_TOWER="google/siglip-so400m-patch14-384"
                MODEL_TYPE="llama3-8b"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision1_5-8b" ]; then
                MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
                MODEL_NAME_OFF="onevision1_5-8b${MODEL_SUFFIX}"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "onevision1_5-4b" ]; then
                MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
                MODEL_NAME_OFF="onevision1_5-4b${MODEL_SUFFIX}"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "idefics-8b" ]; then
                MODEL_NAME="HuggingFaceM4/Idefics3-8B-Llama3"
                MODEL_NAME_OFF="idefics-8b${MODEL_SUFFIX}"
                VERSION="llama"
                VISION_TOWER="google/siglip-so400m-patch14-384"
                MODEL_TYPE="llama3-8b"
                BITS=16
            elif [ "$MODEL_ARCH" == "idefics2-8b" ]; then
                MODEL_NAME="HuggingFaceM4/idefics2-8b"
                MODEL_NAME_OFF="idefics2-8b${MODEL_SUFFIX}"
                VERSION="llama"
                VISION_TOWER="google/siglip-so400m-patch14-384"
                MODEL_TYPE="llama3-8b"
                BITS=16
            elif [ "$MODEL_ARCH" == "openflamingo-9b" ]; then
                MODEL_NAME="openflamingo/openflamingo-9b"
                MODEL_NAME_OFF="openflamingo-9b${MODEL_SUFFIX}"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "otter-9b" ]; then
                MODEL_NAME="luodian/otter-9b-hf"
                MODEL_NAME_OFF="otter-9b${MODEL_SUFFIX}"
                VERSION="otter"
                VISION_TOWER="openai/clip-vit-large-patch14"
                MODEL_TYPE="otter"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen3-8b" ]; then
                MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
                MODEL_NAME_OFF="qwen3-8b${MODEL_SUFFIX}"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen3-4b" ]; then
                MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
                MODEL_NAME_OFF="qwen3-4b${MODEL_SUFFIX}"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            elif [ "$MODEL_ARCH" == "qwen3-2b" ]; then
                MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
                MODEL_NAME_OFF="qwen3-2b${MODEL_SUFFIX}"
                VERSION="None"
                VISION_TOWER="None"
                MODEL_TYPE="None"
                BITS=16
            else
                echo "Undefined setting"
                exit 1
            fi
        
            ZEROSHOT=True
            ENABLE_LORA=False
            USE_MTV=False
            if [ "$METHOD" == "Finetune" ]; then
                ZEROSHOT=False
                ENABLE_LORA=True
                if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                    MODEL_NAME_OFF=FINAL_VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}_finetune_${TRAIN_DATASET_NAME}_${EPOCH}
                elif [ "$USE_PAI" == "True" ]; then
                    MODEL_NAME_OFF=PAIx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}_finetune_${TRAIN_DATASET_NAME}_${EPOCH}
                else
                    MODEL_NAME_OFF=${MODEL_NAME_OFF}_finetune_${TRAIN_DATASET_NAME}_${EPOCH}
                fi
            elif [ "$METHOD" == "Finetune-Mixed" ]; then
                ZEROSHOT=False
                ENABLE_LORA=True
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_finetune_mixed_${TRAIN_DATASET_NAME}_${EPOCH}
                TRAIN_SCENARIO=0
            elif [ "$METHOD" == "mt-icl" ]; then
                ZEROSHOT=False
                ENABLE_LORA=True
                TRAIN_SCENARIO=$(echo "$EVAL_SHOT" | sed 's/[^0-9]//g')
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_finetune_${TRAIN_DATASET_NAME}_${EPOCH}
            elif [ "$METHOD" == "dara" ]; then
                ZEROSHOT=False
                TRAIN_SCENARIO=$(echo "$EVAL_SHOT" | sed 's/[^0-9]//g')
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_dara_${TRAIN_DATASET_NAME}_${TRAIN_SCENARIO}_${EPOCH}_${LR}
                TRAIN_SHOT="original"
            elif [ "$METHOD" == "mimic" ]; then
                ZEROSHOT=False
                TRAIN_SCENARIO=$(echo "$EVAL_SHOT" | sed 's/[^0-9]//g')
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_mimic_${TRAIN_DATASET_NAME}_${TRAIN_SCENARIO}_${EPOCH}_${LR}
            elif [ "$METHOD" == "mimic_static" ]; then
                ZEROSHOT=False
                TRAIN_SCENARIO=4
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_mimic_static_${TRAIN_DATASET_NAME}_${TRAIN_SCENARIO}_${EPOCH}_${LR}
            elif [ "$METHOD" == "mimic_mixed" ]; then
                TRAIN_SHOT="mixed"
                ZEROSHOT=False
                TRAIN_SCENARIO=0
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_mimic_mixed_${TRAIN_DATASET_NAME}_${TRAIN_SCENARIO}_${EPOCH}_${LR}
            elif [ "$METHOD" == "stv" ]; then
                TRAIN_SHOT="mixed"
                ZEROSHOT=False
                # TRAIN_SCENARIO=$(echo "$EVAL_SHOT" | sed 's/[^0-9]//g')
                TRAIN_SCENARIO=0
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_stv_${TRAIN_DATASET_NAME}_${TRAIN_SCENARIO}_${EPOCH}
                ENABLE_LORA=True
            elif [ "$METHOD" == "mtv" ]; then
                USE_MTV=True
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_mtv
                TRAIN_DATASET_NAME=$DATASET_NAME
            elif [ "$METHOD" == "mtv2" ]; then
                USE_MTV=True
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_mtv2
                TRAIN_DATASET_NAME=$DATASET_NAME
            elif [ "$METHOD" == "fv" ]; then
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_fv
                TRAIN_DATASET_NAME=$DATASET_NAME
            elif [ "$METHOD" == "grpo" ]; then
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_grpo_${REWARD_TAG}_RW${RW_STR}_${EPOCH}_${LR}${GRPO_METHOD_TAG}
                ZEROSHOT=False
                ENABLE_LORA=True
            elif [ "$METHOD" == "lrv" ]; then
                MODEL_NAME_OFF=${MODEL_NAME_OFF}_lrv
                ZEROSHOT=False
                ENABLE_LORA=True
            elif [ "$METHOD" == "volcano" ]; then
                ZEROSHOT=False
                ENABLE_LORA=True
            else
                TRAIN_DATASET_NAME=$DATASET_NAME
            fi

            if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                MODEL_NAME_OFF=FINAL_VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}
            elif [ "$USE_PAI" == "True" ]; then
                MODEL_NAME_OFF=PAIx${VIS_TOKEN_AMPLIFIER}_${MODEL_NAME_OFF}
            fi

            if [[ "$DATASET_NAME" == *"_new"* ]]; then
                TRAIN_DATASET_NAME="${DATASET_NAME//_new/}"
            fi

            if [ "$USE_ICOT" == "True" ]; then
                MODEL_NAME_OFF=ICOT_VER${COTPROMPT_VER}_${MODEL_NAME_OFF}
            elif [ "$USE_COTPROMPT" == "True" ]; then
                MODEL_NAME_OFF=COTPROMPT_VER${COTPROMPT_VER}_${MODEL_NAME_OFF}
            fi
            for ((ind2=0; ind2<${#ORDERS[@]}; ind2++)); do
                ORDER=${ORDERS[$ind2]}
                
                if [ "$ORDER" == "lure" ]; then
                    for s in "${SCENARIOS[@]}"; do
                        SCENARIOS+=($((4*s-2)) $((4*s-1)) $((4*s)))
                    done
                fi

                for ((ind=0; ind<${#SCENARIOS[@]}; ind++)); do
                    SCENARIO=${SCENARIOS[$ind]}
                    MTV_PATH="./MTV/storage/${MODEL_ARCH}/${DATASET_NAME}/mtv_${SCENARIO}.pt"
                    MTV_ACT_PATH="./MTV/storage/${MODEL_ARCH}/${DATASET_NAME}/mtv_activation_${SCENARIO}.pt"
                    if [ "$METHOD" == "dara" ]; then
                        MODE=dara
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}_${LR}
                        ATTEN_IMPLE="eager"
                    elif [ "$METHOD" == "mimic" ] ||  [ "$METHOD" == "mimic_static" ] ||  [ "$METHOD" == "mimic_mixed" ]; then
                        MODE=mimic
                        NOTE=mimic_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}_${LR}
                        ATTEN_IMPLE="flash_attention_2"
                        if [ "$MODEL_ARCH" == "internvl3_5-8b" ]; then
                            ATTEN_IMPLE="flash_attention_2"
                        fi
                    elif [ "$METHOD" == "only" ]; then
                        MODE=only
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EVAL_SHOT}_scen${TRAIN_SCENARIO}
                        ATTEN_IMPLE="eager"
                    elif [ "$METHOD" == "mtv" ]; then
                        NOTE=${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EVAL_SHOT}_${ORDER}_scen_${SCENARIO}_${TRAIN_SCENARIO}_mtv_${USE_MTV}
                        ATTEN_IMPLE="flash_attention_2"
                    elif [ "$METHOD" == "mtv2" ]; then
                        MTV_PATH="./MTV/storage/${MODEL_ARCH}/${DATASET_NAME}/mtv_${DATASET_NAME}.pt"
                        MTV_ACT_PATH="./MTV/storage/${MODEL_ARCH}/${DATASET_NAME}/mtv_activation_${DATASET_NAME}.pt"
                        NOTE=${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EVAL_SHOT}_${ORDER}_scen_${SCENARIO}_${TRAIN_SCENARIO}_mtv2_${USE_MTV}
                        ATTEN_IMPLE="flash_attention_2"
                    elif [ "$METHOD" == "fv" ]; then
                        MODE=fv
                        NOTE=${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EVAL_SHOT}_${ORDER}_scen_${SCENARIO}_fv
                        ATTEN_IMPLE="flash_attention_2"
                    elif [ "$METHOD" == "stv" ]; then
                        MODE=stv
                        NOTE=stv_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${TRAIN_SHOT}_scen0
                        ATTEN_IMPLE="flash_attention_2"
                    elif [ "$METHOD" == "lrv" ]; then
                        NOTE=${MODE}_${MODEL_ARCH}_lrv_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}
                        ATTEN_IMPLE="flash_attention_2"
                    elif [ "$METHOD" == "volcano" ]; then
                        echo "volcano method"
                        NOTE=${MODE}_${MODEL_ARCH}_volcano_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}
                        ATTEN_IMPLE="flash_attention_2"
                    elif [ "$METHOD" == "mt-icl" ]; then
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}
                        ATTEN_IMPLE="sdpa"
                        if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                            NOTE=VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}
                            ATTEN_IMPLE="eager"
                        fi
                    elif [ "$METHOD" == "Finetune" ] || [ "$METHOD" == "Finetune-Mixed" ]; then
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}
                        ATTEN_IMPLE="flash_attention_2"
                        # if [ "$MODEL_ARCH" == "internvl3_5-8b" ]; then
                        #     ATTEN_IMPLE="eager"
                        # fi
                        if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                            NOTE=VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EPOCH}_${TRAIN_SHOT}_scen${TRAIN_SCENARIO}
                            ATTEN_IMPLE="eager"
                        fi
                    elif [ "$METHOD" == "grpo" ]; then
                        # Must equal train_GRPO_VLM.sh NOTE (drives ./client_states_<NOTE>/).
                        if [[ -n "$GRPO_NOTE" ]]; then
                            NOTE="$GRPO_NOTE"
                        else
                            NOTE="$(build_grpo_note "$MODEL_ARCH")"
                        fi
                        ATTEN_IMPLE="sdpa"
                        MODE=grpo
                    else
                        NOTE=${MODE}_${MODEL_ARCH}_${TRAIN_DATASET_NAME}_${EVAL_SHOT}_${ORDER}_scen${SCENARIO}
                        ATTEN_IMPLE="flash_attention_2"  
                        if [ "$MODEL_ARCH" == "internvl3_5-1b" ] || [ "$MODEL_ARCH" == "internvl3_5-2b" ] || [ "$MODEL_ARCH" == "internvl3_5-4b" ] || [ "$MODEL_ARCH" == "internvl3_5-8b" ] || [ "$MODEL_ARCH" == "internvl3_5-14b" ]; then
                            ATTEN_IMPLE="flash_attention_2"  
                        fi
                        if [ "$USE_VIS_TOKEN_AMPLIFIER" == "True" ]; then
                            NOTE=VIS_AMPLIFIERx${VIS_TOKEN_AMPLIFIER}_${NOTE}
                            ATTEN_IMPLE="eager"
                        elif [ "$USE_PAI" == "True" ]; then
                            NOTE=PAIx${VIS_TOKEN_AMPLIFIER}_${NOTE}
                            ATTEN_IMPLE="eager"
                        fi

                        if [ "$USE_ICOT" == "True" ]; then
                            NOTE=ICOT_${COTPROMPT_VER}_${NOTE}
                            ATTEN_IMPLE="eager"
                        elif [ "$USE_COTPROMPT" == "True" ]; then
                            NOTE=COTPROMPT_VER${COTPROMPT_VER}_${NOTE}
                        fi
                        if [ "$VARIANT" == "sparc" ]; then
                            ATTEN_IMPLE="eager"
                        fi
                    fi

                    for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
                        if [[ "$METHOD" == "grpo" ]]; then
                            CHECKPOINT_PATH="./client_states_${NOTE}/server_model_round$((ROUND_TO_EVALS[index] - 1)).pth"
                            echo "[GRPO eval] reward=${REWARD} checkpoint=${CHECKPOINT_PATH}"
                            if [[ ! -f "$CHECKPOINT_PATH" ]]; then
                                echo "Error: GRPO checkpoint not found: $CHECKPOINT_PATH" >&2
                                echo "Set GRPO_NOTE to the exact NOTE used for training if the run used different settings." >&2
                                exit 2
                            fi
                        fi
                        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                        CUDA_VISIBLE_DEVICES=$GPU python ${EVAL_SCRIPT} \
                            --is_eval True \
                            "${EXTRA_ARGS[@]}" \
                            --model_name_or_path $MODEL_NAME \
                            --model_name_for_dataarg $MODEL_NAME \
                            --model_name $MODEL_NAME_OFF \
                            --model_type $MODEL_TYPE \
                            --version $VERSION \
                            --scenario $SCENARIO \
                            --train_scenario $TRAIN_SCENARIO \
                            --num_rounds $NUM_ROUNDS \
                            --num_tasks $NUM_TASKS \
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
                            --zeroshot $ZEROSHOT \
                            --lora_enable $ENABLE_LORA \
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
                            --dataset_name $DATASET_NAME \
                            --setup "" \
                            --ordering $ORDER \
                            --shot $TRAIN_SHOT \
                            --eval_shot $EVAL_SHOT \
                            --attn_impl $ATTEN_IMPLE \
                            --use_mtv $USE_MTV \
                            --mtv_path $MTV_PATH \
                            --use_pai $USE_PAI \
                            --mtv_activation_path $MTV_ACT_PATH \
                            --use_vis_token_amplifier $USE_VIS_TOKEN_AMPLIFIER \
                            --vis_token_amplifier $VIS_TOKEN_AMPLIFIER \
                            --use_cotprompt $USE_COTPROMPT \
                            --cotprompt_ver $COTPROMPT_VER \
                            --use_icot $USE_ICOT \
                            --reward_weight "${REWARD_WEIGHT[@]}" \
                            --caprl_eval "$CAPRL_EVAL" \
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
                            --output_dir "./nohup" #> ./nohup2/eval3_${NOTE}.log 2>&1 &
                    done
                done
            done
        done
    done
done
