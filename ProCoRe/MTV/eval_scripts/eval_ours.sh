MODEL="qwen3-8b" # onevision1_5-8b, qwen3-8b, internvl3_5-8b, idefics-8b, otter-9b
DATASET_NAME="pororov4"
ORDERING="random"
SCENARIO=1

STORAGE_DIR="MTV/storage/${MODEL}/${DATASET_NAME}"
STORAGE_JSON_DIR="MTV/storage_json/${MODEL}/${DATASET_NAME}"

mkdir -p "${STORAGE_DIR}"
mkdir -p "${STORAGE_JSON_DIR}"

if [[ "$DATASET_NAME" == "pororo" || "$DATASET_NAME" == "flintstones" || "$DATASET_NAME" == "storystream" || "$DATASET_NAME" == "story_visual_ICL" ]]; then
    NUM_EXAMPLE=50
    SHOT="mixed"
elif [[ "$DATASET_NAME" == "mic" || "$DATASET_NAME" == "vlicl" || "$DATASET_NAME" == "vliclv2" || "$DATASET_NAME" == "truemicl" || "$DATASET_NAME" == "isekai" || "$DATASET_NAME" == "isekai_4choice" || "$DATASET_NAME" == "SMMILE" || "$DATASET_NAME" == "illusionvqa-comp" || "$DATASET_NAME" == "idk" || "$DATASET_NAME" == "stanford10000" || "$DATASET_NAME" == "movie101" || "$DATASET_NAME" == "moviegraphs" || "$DATASET_NAME" == "memor" || "$DATASET_NAME" == "pororov2" || "$DATASET_NAME" == "pororov3" || "$DATASET_NAME" == "story_visual_ICLv2" || "$DATASET_NAME" == "storystreamv2" ]]; then
    NUM_EXAMPLE=100
    SHOT="4-shot"
elif [[ "$DATASET_NAME" == "stanford10000_5" ]]; then
    NUM_EXAMPLE=100
    SHOT="5-shot"
else
    echo "Warning: Unknown DATASET_NAME. Defaulting to 100/mixed."
    NUM_EXAMPLE=100
    SHOT="mixed"
fi

SCENARIO_FILE=./scenarios/${SHOT}/${DATASET_NAME}/${ORDERING}/scenario-${SCENARIO}.json

read PARSED_DATASET PARSED_SUBSET <<< $(python3 -c "import json, sys; data=json.load(open(sys.argv[1]))[0]['datasets'][0]; print(data['dataset'], data['subset_id'])" "${SCENARIO_FILE}")

DATALISTS_PATH="datalists/${PARSED_DATASET}/dataset-${PARSED_SUBSET}.json"

#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 -m MTV.mtv_eval \
    --model_name ${MODEL} \
    --data_name ${DATASET_NAME} \
    --train_path ${DATALISTS_PATH} \
    --val_path ${DATALISTS_PATH} \
    --num_example ${NUM_EXAMPLE} \
    --num_shot 0 \
    --max_token 512 \
    --eval_num_shot 0 \
    --bernoullis_path ${STORAGE_DIR}/mtv_${SCENARIO}.pt \
    --activation_path ${STORAGE_DIR}/mtv_activation_${SCENARIO}.pt \
    --is_eval True \
    --result_folder ./ \
    --cur_mode both \
    --experiment_name ${DATASET_NAME}
