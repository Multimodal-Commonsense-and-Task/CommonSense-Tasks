#!/bin/bash

MODEL="llama3"
METHOD="None" # None, abstract, detailed
DATASET="dataset/LTF_TEST.jsonl"
OUTPUT_DIR="outputs"
BATCH=2

Generate outputs for testing
CUDA_VISIBLE_DEVICES=7 python test.py \
  --model $MODEL \
  --mode inference \
  --method $METHOD \
  --dataset $DATASET \
  --batch $BATCH \
  --output_dir $OUTPUT_DIR

# Evaluate outputs
CUDA_VISIBLE_DEVICES=7 python test.py \
  --model $MODEL \
  --mode evaluation \
  --method $METHOD \
  --dataset $DATASET \
  --batch $BATCH \
  --output_dir $OUTPUT_DIR