#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -m mtv_eval \
    --model_name Qwen-VL \
    --data_name okvqa \
    --train_path ./data/okvqa/okvqa_train.jsonl \
    --val_path ./data/okvqa/okvqa_val.jsonl \
    --num_example 100 \
    --num_shot 4 \
    --max_token 20 \
    --eval_num_shot 0 \
    --bernoullis_path ./storage/okvqa_mtv.pt \
    --activation_path ./storage/okvqa_mtv_activation.pt \
    --is_eval True \
    --result_folder ./ \
    --cur_mode interv \
    --experiment_name temp