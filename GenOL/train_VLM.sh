#/bin/bash
# CIL CONFIG
#bongard_openworld_ma_num9_iter1
NOTE="Bongard_OpenWorld_generated_RMD_diversified_more_num7_iter1_meminf" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="VLM"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b
RND_SEED=1

# fed args
DATASET="Bongard-OpenWorld"
DATA_TYPE="generated_RMD_diversified_more" #ma, generaetd, web, generated_RMD, generated_diversified_sdxl_more, generated_RMD_diversified_more
NUM_SET=7 # 5 - support set : 4 (2 positive, 2 negative) + 1 query, choice = [5, 7, 9]
MODEL_MAX_LEN=10000
MEMORY_SIZE=100000
NUM_ITER=1
BATCHSIZE=4
LR=5e-5
MM_PROJECTOR_LR=5e-5
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="constant" #cosine
WARMUP_RATIO=0.03 # SHOULD BE 0.03 / NUM_ROUNDS

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="./llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="./clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16

elif [ "$MODEL_ARCH" == "bunny_3b" ]; then
    MODEL_NAME="BAAI/Bunny-v1_0-3B"
    VERSION="bunny"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="phi-2"
    BITS=16
elif [ "$MODEL_ARCH" == "bunny_8b" ]; then
    MODEL_NAME="BAAI/Bunny-Llama-3-8B-V"
    VERSION="llama"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="llama3-8b"
    BITS=8
else
    echo "Undefined setting"
    exit 1
fi
# --master_port 29500
    nohup deepspeed --master_port 29610 \
    --include localhost:4,5,6,7 \
    main_new_llava_trainer.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --model_max_length $MODEL_MAX_LEN \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --dataset $DATASET \
    --num_set $NUM_SET \
    --data_type $DATA_TYPE \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 2 \
    --num_iter $NUM_ITER \
    --note $NOTE \
    --memory_size $MEMORY_SIZE \
    --output_dir "./results/test/" & # > ./nohup/fedavg_llava_sc12_lr5e-5_bs16_itr100_constant_nodist.log 2>&1 &

# --eval_period $EVAL_PERIOD
#
