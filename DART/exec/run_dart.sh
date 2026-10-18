#!/bin/bash
# Quick example script for Domain ARiThmetic (DART).

set -e

BASE_POLICY=/path/to/base_policy
POLICY_SRC=/path/to/policy_src
POLICY_TGT=/path/to/policy_tgt

CONFIG=pi05_libero_oneshotft
BASE_CONFIG=pi05_libero
SCALING_COEF=0.8

OUTPUT_BASE=/path/to/output
OUTPUT_DIR=${OUTPUT_BASE}/DART_${SCALING_COEF}

echo "=========================================="
echo "Running: DART (scaling_coef=${SCALING_COEF})"
echo "=========================================="

uv run -m domain_arithmetic.dart \
    --cfg.base_policy.dir ${BASE_POLICY} \
    --cfg.base_policy.config ${BASE_CONFIG} \
    --cfg.policy_src.dir ${POLICY_SRC} \
    --cfg.policy_src.config ${CONFIG} \
    --cfg.policy_tgt.dir ${POLICY_TGT} \
    --cfg.policy_tgt.config ${CONFIG} \
    --cfg.scaling_coef ${SCALING_COEF} \
    --cfg.output_dir ${OUTPUT_DIR} \
    --cfg.overwrite

echo "[DONE] DART -> ${OUTPUT_DIR}"
