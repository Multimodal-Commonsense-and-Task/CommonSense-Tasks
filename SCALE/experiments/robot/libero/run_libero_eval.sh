#!/bin/bash
# Convenience wrapper around run_libero_eval.py.
#
# Usage:
#   bash experiments/robot/libero/run_libero_eval.sh <GPU_ID> <TASK_SUITE> <DECODING_MODE>
#
#   GPU_ID        : CUDA device index (default: 0)
#   TASK_SUITE    : libero_spatial | libero_object | libero_goal | libero_10   (default: libero_spatial)
#   DECODING_MODE : greedy | temp | topk | topp | scale                        (default: scale)
#
# Examples:
#   bash experiments/robot/libero/run_libero_eval.sh 0 libero_10      scale     # SCALE (paper defaults from configs/scale.yaml)
#   bash experiments/robot/libero/run_libero_eval.sh 0 libero_spatial greedy    # greedy baseline
#   bash experiments/robot/libero/run_libero_eval.sh 1 libero_object  topk      # top-k (k=40, t=0.7 -> see flags below)
set -e

GPU_ID=${1:-0}
TASK_SUITE=${2:-libero_spatial}
DECODING_MODE=${3:-scale}

# Map task suite -> official OpenVLA finetuned checkpoint (HF Hub IDs)
case "$TASK_SUITE" in
    libero_spatial) CHECKPOINT="openvla/openvla-7b-finetuned-libero-spatial" ;;
    libero_object)  CHECKPOINT="openvla/openvla-7b-finetuned-libero-object"  ;;
    libero_goal)    CHECKPOINT="openvla/openvla-7b-finetuned-libero-goal"    ;;
    libero_10)      CHECKPOINT="openvla/openvla-7b-finetuned-libero-10"      ;;
    *) echo "Unknown task suite: $TASK_SUITE" && exit 1 ;;
esac

# Per-mode baseline flags (paper baseline configurations; SCALE reads configs/scale.yaml).
EXTRA_FLAGS=""
case "$DECODING_MODE" in
    greedy) ;;
    temp)   EXTRA_FLAGS="--temperature 1.0" ;;                 # Sampling (t=1.0)
    topk)   EXTRA_FLAGS="--top_k 40 --temperature 0.7" ;;      # Top-k (k=40, t=0.7)
    topp)   EXTRA_FLAGS="--top_p 0.9" ;;                       # Top-p (p=0.9)
    scale)  ;;                                                 # SCALE defaults from configs/scale.yaml
    *) echo "Unknown decoding mode: $DECODING_MODE" && exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite "${TASK_SUITE}" \
    --decoding_mode "${DECODING_MODE}" \
    ${EXTRA_FLAGS}
