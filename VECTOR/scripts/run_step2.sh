#!/bin/bash
# ==============================================================================
# Step 2: Task-specific inference using descriptions from Step 1
#
# Usage:
#   bash scripts/run_step2.sh --gpus 0,1,2,3 --task_id 1 --level 1
#   bash scripts/run_step2.sh --gpus 0,1,2,3 --task_id 3 --level 1 --variant single
#   bash scripts/run_step2.sh --gpus 0,1,2,3 --task_id 5 --level 1
#
# Note: Step 1 must be completed first.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Defaults ----
CKPT="${PROJECT_ROOT}/checkpoints/llava-onevision-qwen2-7b-ov-multi-event"
GPU_STR=""
TASK_ID=""
LEVEL=""
VARIANT=""

# ---- Parse Arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)     GPU_STR="$2";  shift 2 ;;
        --task_id)  TASK_ID="$2";  shift 2 ;;
        --level)    LEVEL="$2";    shift 2 ;;
        --variant)  VARIANT="$2";  shift 2 ;;
        --ckpt)     CKPT="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$GPU_STR" ] || [ -z "$TASK_ID" ] || [ -z "$LEVEL" ]; then
    echo "Usage: $0 --gpus 0,1,2,3 --task_id <1-5> --level <1|2> [--variant single|double|triple] [--ckpt PATH]"
    exit 1
fi

# ---- GPU setup ----
IFS=',' read -ra GPULIST <<< "$GPU_STR"
CHUNKS=${#GPULIST[@]}

# ---- Fixed configuration ----
NFRAME=32
CONV_MODE=qwen_1_5
POOL_STRIDE=2
TEMPERATURE=0
OVERWRITE=True
VIDEO_DIR="${PROJECT_ROOT}/kinetics-dataset/k700-2020/val"
VNAME2CLS="${PROJECT_ROOT}/kinetics-dataset/kinetics_jsonl/k700_vname2cls.json"
CKPT_NAME=$(basename "$CKPT")
STEP1_BASE="${PROJECT_ROOT}/results/${CKPT_NAME}/step1"
RESULT_BASE="${PROJECT_ROOT}/results/${CKPT_NAME}/step2"

export TOKENIZERS_PARALLELISM=false

# ---- Build run list: (step1_input, output_tag, question_type, extra_flags) ----
declare -a STEP1_FILES=()
declare -a OUTPUT_TAGS=()
declare -a Q_TYPES=()
declare -a EXTRA_FLAGS=()

case $TASK_ID in
    1)
        TAG="task1_L${LEVEL}"
        STEP1_FILES=("${STEP1_BASE}/${TAG}.jsonl")
        OUTPUT_TAGS=("${TAG}")
        Q_TYPES=("action_summary")
        EXTRA_FLAGS=("--add_class_prompt --add_naction_helper_strict --add_format_prompt")
        ;;
    2)
        TAG="task2_L${LEVEL}"
        STEP1_FILES=("${STEP1_BASE}/${TAG}.jsonl")
        OUTPUT_TAGS=("${TAG}")
        Q_TYPES=("action_summary_between")
        EXTRA_FLAGS=("--add_class_prompt --add_naction_helper_strict --add_format_prompt")
        ;;
    3)
        if [ -n "$VARIANT" ]; then
            VARIANTS=("$VARIANT")
        else
            VARIANTS=(single double triple)
        fi
        for V in "${VARIANTS[@]}"; do
            TAG="task3_L${LEVEL}_${V}"
            STEP1_FILES+=("${STEP1_BASE}/${TAG}.jsonl")
            OUTPUT_TAGS+=("${TAG}")
            Q_TYPES+=("action_niah_${V}")
            EXTRA_FLAGS+=("--add_format_prompt")
        done
        ;;
    4)
        TAG="task4_L${LEVEL}"
        STEP1_FILES=("${STEP1_BASE}/${TAG}.jsonl")
        OUTPUT_TAGS=("${TAG}")
        Q_TYPES=("action_anomaly_domain_loc")
        EXTRA_FLAGS=("--add_format_prompt")
        ;;
    5)
        # L1: shorter patterns (ABABAB, ABCABC), L2: longer patterns (ABABABAB, ABCABCABC)
        if [ "$LEVEL" = "1" ]; then
            PATTERNS=(ABABAB ABCABC)
        else
            PATTERNS=(ABABABAB ABCABCABC)
        fi
        for P in "${PATTERNS[@]}"; do
            TAG="task5_L${LEVEL}_${P}"
            STEP1_FILES+=("${STEP1_BASE}/${TAG}.jsonl")
            OUTPUT_TAGS+=("${TAG}")
            Q_TYPES+=("action_anomaly_patternbreak_loc")
            EXTRA_FLAGS+=("--add_format_prompt")
        done
        ;;
    *)
        echo "Error: task_id must be 1-5, got: $TASK_ID"
        exit 1
        ;;
esac

# ---- Run inference ----
echo "============================================"
echo "Step 2: Task-Specific Inference"
echo "============================================"
echo "  Checkpoint: ${CKPT}"
echo "  Task:       ${TASK_ID}"
echo "  Level:      ${LEVEL}"
echo "  GPUs:       ${GPULIST[*]} (${CHUNKS} chunks)"
echo "  Step 1 dir: ${STEP1_BASE}"
echo "  Output:     ${RESULT_BASE}"
echo "============================================"

for i in "${!STEP1_FILES[@]}"; do
    DATA_PATH="${STEP1_FILES[$i]}"
    TAG="${OUTPUT_TAGS[$i]}"
    QTYPE="${Q_TYPES[$i]}"
    FLAGS="${EXTRA_FLAGS[$i]}"
    OUTPUT_DIR="${RESULT_BASE}/${TAG}"
    OUTPUT_FILE="${OUTPUT_DIR}.jsonl"

    if [ ! -f "$DATA_PATH" ]; then
        echo "WARNING: Step 1 output not found: ${DATA_PATH}"
        echo "         Run Step 1 first: bash scripts/run_step1.sh --gpus ${GPU_STR} --task_id ${TASK_ID} --level ${LEVEL}"
        continue
    fi

    echo ""
    echo "--- ${TAG} (${QTYPE}) ---"
    echo "  Input:  ${DATA_PATH}"
    echo "  Output: ${OUTPUT_DIR}"

    mkdir -p "${OUTPUT_DIR}"

    # Launch parallel inference across GPUs
    for IDX in $(seq 0 $((CHUNKS - 1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 "${PROJECT_ROOT}/eval/infer_wdesc.py" \
            --model-path "$CKPT" \
            --video_path "$VIDEO_DIR" \
            --gt_file "$DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --vname2cls "$VNAME2CLS" \
            --question_type "$QTYPE" \
            --conv-mode "$CONV_MODE" \
            --for_get_frames_num "$NFRAME" \
            --mm_spatial_pool_stride "$POOL_STRIDE" \
            --temperature "$TEMPERATURE" \
            --overwrite "$OVERWRITE" \
            --overwrite_infercfg true \
            ${FLAGS} \
            --num-chunks "$CHUNKS" \
            --chunk-idx "$IDX" &
    done
    wait

    # Concatenate chunk outputs
    > "$OUTPUT_FILE"
    for IDX in $(seq 0 $((CHUNKS - 1))); do
        cat "${OUTPUT_DIR}/${CHUNKS}_${IDX}.jsonl" >> "$OUTPUT_FILE"
    done
    echo "  Combined: ${OUTPUT_FILE} ($(wc -l < "$OUTPUT_FILE") lines)"

    # Compute accuracy
    echo ""
    echo "--- Accuracy: ${TAG} (${QTYPE}) ---"
    python3 "${PROJECT_ROOT}/eval/compute_accuracy.py" "$OUTPUT_FILE" "$QTYPE"
done

echo ""
echo "============================================"
echo "Step 2 Complete"
echo "============================================"
