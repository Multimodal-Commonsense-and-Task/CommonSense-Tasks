#!/usr/bin/env bash

# Example usage:
#   GPU_ID=0 TASK_SUITE_NAME=libero_spatial CAMERA_VIEW=medium bash exec/eval.sh

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Select which host GPU this evaluation uses. The default port and compose
# project name are derived from this so multiple evals can run side by side.
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-$((9012 + GPU_ID))}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-libero_gpu${GPU_ID}}"

# Docker exposes DOCKER_GPU_ID from the host. Inside the container, that exposed
# GPU is usually visible as logical device 0, which MuJoCo EGL should use.
DOCKER_GPU_ID="${DOCKER_GPU_ID:-${GPU_ID}}"
CONTAINER_GPU_ID="${CONTAINER_GPU_ID:-0}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

# Policy server config. Leave CHECKPOINT_DIR empty to use the default LIBERO
# checkpoint from scripts/serve_policy.py.
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

# Convert repo-local checkpoint paths to the path seen inside the container,
# because the repository root is mounted at /app by examples/libero-view/compose.yml.
SERVER_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
if [[ -n "${CHECKPOINT_DIR}" && "${CHECKPOINT_DIR}" != gs://* ]]; then
  if [[ "${CHECKPOINT_DIR}" == "${REPO_ROOT}"/* ]]; then
    SERVER_CHECKPOINT_DIR="/app/${CHECKPOINT_DIR#"${REPO_ROOT}/"}"
  elif [[ "${CHECKPOINT_DIR}" != /* ]]; then
    SERVER_CHECKPOINT_DIR="/app/${CHECKPOINT_DIR}"
  fi
fi

# LIBERO client options.
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
CAMERA_VIEW="${CAMERA_VIEW:-original}"
PERTURB="${PERTURB:-}"

# Arguments consumed by scripts/serve_policy.py inside the openpi_server
# container.
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  export SERVER_ARGS="--env LIBERO --port ${PORT} policy:checkpoint --policy.config ${POLICY_CONFIG} --policy.dir ${SERVER_CHECKPOINT_DIR}"
else
  export SERVER_ARGS="--env LIBERO --port ${PORT}"
fi

# Arguments consumed by examples/libero-view/main.py inside the runtime container.
export CLIENT_ARGS="--args.host 127.0.0.1 --args.port ${PORT} --args.task-suite-name ${TASK_SUITE_NAME} --args.camera-view ${CAMERA_VIEW}"
if [[ -n "${PERTURB}" ]]; then
  CLIENT_ARGS="${CLIENT_ARGS} --args.perturb ${PERTURB}"
fi

# Variables consumed by examples/libero-view/compose.yml.
export COMPOSE_PROJECT_NAME
export DOCKER_GPU_ID
export CONTAINER_GPU_ID
export MUJOCO_GL

# Build images if needed, start the policy server and LIBERO runtime, and return
# the runtime container's exit code.
docker compose -f examples/libero-view/compose.yml up --build --abort-on-container-exit --exit-code-from runtime
