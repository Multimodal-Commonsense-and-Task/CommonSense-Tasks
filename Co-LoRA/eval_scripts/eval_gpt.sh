#!/bin/bash

base_dir="zeroshot/llama3_8b_fedaya_topic"

# Associate each integer value with a list of corresponding dataset names
declare -A dataset_map
# List of datasets that are not multi-choice
dataset_map[0]="Fed-aya_topic-0 Fed-aya_topic-1 Fed-aya_topic-2 Fed-aya_topic-3"
dataset_map[1]="Fed-aya_topic-10 Fed-aya_topic-11 Fed-aya_topic-12 Fed-aya_topic-13"
dataset_map[2]="Fed-aya_topic-20 Fed-aya_topic-21 Fed-aya_topic-22 Fed-aya_topic-23"
dataset_map[3]="Fed-aya_topic-30 Fed-aya_topic-31 Fed-aya_topic-32 Fed-aya_topic-33"
dataset_map[4]="Fed-aya_topic-40 Fed-aya_topic-41 Fed-aya_topic-42 Fed-aya_topic-43"
dataset_map[5]="Fed-aya_topic-50 Fed-aya_topic-51 Fed-aya_topic-52 Fed-aya_topic-53"
dataset_map[6]="Fed-aya_topic-60 Fed-aya_topic-61 Fed-aya_topic-62 Fed-aya_topic-63"
dataset_map[7]="Fed-aya_topic-70 Fed-aya_topic-71 Fed-aya_topic-72 Fed-aya_topic-73"

ROUND_TO_EVALS=(20)

# Iterate over the keys (integer values) in the associative array
for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
    for i in "${!dataset_map[@]}"; do
        datasets=(${dataset_map[$i]})
        for dataset in "${datasets[@]}"; do
            input_file="./eval_results/${base_dir}/client${i}_round${ROUND_TO_EVALS[$index]}_${dataset}.json"
            output_file="./eval_results_gpt/${base_dir}/client${i}_round${ROUND_TO_EVALS[$index]}_${dataset}.jsonl"
            
            echo "Processing ${dataset} (client ${i})..."
            OPENAI_API_KEY="" \
            python -m eval_scripts.eval_gpt_aya -r "$input_file" -o "$output_file" --random_seed 42 > gpteval.out 2>&1 &
        done
    done
done
echo "All datasets processed."
