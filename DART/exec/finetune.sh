#!/usr/bin/env bash

# Example usage:
#   DATA_REPO_ID=/path/to/libero_viewpoints_5demo CHECKPOINT_BASE_DIR=/path/to/checkpoints CAMERA_VIEW=medium bash exec/finetune.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# WandB settings. Do not hard-code WANDB_API_KEY here; export it in your shell if
# needed before running this script.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-domain-arithmetic}"

# GPU / JAX settings.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# Training config and data settings.
CONFIG="${CONFIG:-pi05_libero_oneshotft}"
DATA_REPO_ID="${DATA_REPO_ID:?Set DATA_REPO_ID to your LeRobot LIBERO dataset path or repo id}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:?Set CHECKPOINT_BASE_DIR to the output checkpoint directory}"

CAMERA_VIEW="${CAMERA_VIEW:-original}"

# Training hyperparameters.
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-1000}"

echo "CONFIG=${CONFIG}"
echo "DATA_REPO_ID=${DATA_REPO_ID}"
echo "CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"
echo "CAMERA_VIEW=${CAMERA_VIEW}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "===== ONE-SHOT FINETUNE LIBERO ====="
echo "camera_view=${CAMERA_VIEW}"
echo "====================================="

uv run scripts/train.py "${CONFIG}" \
  --project-name="${CAMERA_VIEW}" \
  --batch-size="${BATCH_SIZE}" \
  --data.repo-id="${DATA_REPO_ID}" \
  --data.camera-view="${CAMERA_VIEW}" \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
  --exp-name="${CAMERA_VIEW}" \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  --save-interval="${SAVE_INTERVAL}" \
  --keep-period="${KEEP_PERIOD}"
