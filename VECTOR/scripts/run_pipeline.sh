#!/bin/bash
# ==============================================================================
# Unified evaluation pipeline
#
# Modes:
#   --llava-onevision  (default) Direct 1-step inference with LLaVA-OneVision
#   --mecot            2-step MeCoT: Step 1 (describe) → Step 2 (task-specific)
#
# Usage:
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 1 --level 1
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 1 --level 1 --llava-onevision
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 1 --level 1 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 3 --level 1 --variant single --mecot
#
# Run all tasks (LLaVA-OneVision, default):
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 1 --level 1
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 1 --level 2
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 2 --level 1
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 2 --level 2
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 3 --level 1
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 3 --level 2
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 4 --level 1
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 4 --level 2
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 5 --level 1
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 5 --level 2
#
# Run all tasks (MeCoT, 2-step):
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 1 --level 1 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 1 --level 2 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 2 --level 1 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 2 --level 2 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 3 --level 1 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 3 --level 2 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 4 --level 1 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 4 --level 2 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 5 --level 1 --mecot
#   bash scripts/run_pipeline.sh --gpus 0,1,2,3 --task_id 5 --level 2 --mecot
#
# Arguments:
#   --gpus      Required. Comma-separated GPU IDs (e.g., 0,1,2,3)
#   --task_id   Required. Task ID (1-5)
#   --level     Required. Level (1 or 2)
#   --variant   Task 3 only. single/double/triple (default: run all)
#   --ckpt      Optional. Checkpoint path (overrides mode default)
#   --mecot     Use 2-step MeCoT pipeline (describe → infer with description)
#   --llava-onevision  Use direct 1-step inference (default)
#
# Tasks:
#   1: Action Summary (Recognition)
#   2: Relative Query (Between)
#   3: NIAH - Needle in a Haystack (single/double/triple)
#   4: Anomaly Domain Detection
#   5: Anomaly Pattern-Break Detection (L1: ABABAB+ABCABC, L2: ABABABAB+ABCABCABC)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Detect mode ----
MODE="llava-onevision"  # default
FORWARD_ARGS=()

for arg in "$@"; do
    case $arg in
        --mecot)
            MODE="mecot"
            ;;
        --llava-onevision)
            MODE="llava-onevision"
            ;;
        *)
            FORWARD_ARGS+=("$arg")
            ;;
    esac
done

echo "======================================================"
echo "  Running Evaluation Pipeline"
echo "  Mode: ${MODE}"
echo "  Args: ${FORWARD_ARGS[*]}"
echo "======================================================"

if [ "$MODE" = "mecot" ]; then
    echo ""
    echo "[1/2] Running Step 1: Video Description Generation..."
    bash "${SCRIPT_DIR}/run_step1.sh" "${FORWARD_ARGS[@]}"

    echo ""
    echo "[2/2] Running Step 2: Task-Specific Inference..."
    bash "${SCRIPT_DIR}/run_step2.sh" "${FORWARD_ARGS[@]}"
else
    echo ""
    echo "[1/1] Running Direct Inference..."
    bash "${SCRIPT_DIR}/run_direct.sh" "${FORWARD_ARGS[@]}"
fi

echo ""
echo "======================================================"
echo "  Pipeline Complete (${MODE})"
echo "======================================================"
