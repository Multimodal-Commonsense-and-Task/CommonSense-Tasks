# export PYTHONPATH=$PYTHONPATH:./

# For cluster
# export ADDR=$1
# run_args="--nproc_per_node 8 \
#           --master_addr $ADDR \
#           --node_rank $RANK \
#           --master_port $MASTER_PORT \
#           --nnodes $WORLD_SIZE"
# For local

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# run_args="--nproc_per_node 8 --master_port 29511"

export CUDA_VISIBLE_DEVICES=0
run_args="--nproc_per_node 1 --master_port 29511"

USE_FP16=True

# Batch size per GPU
BS=112
# BS=4

# Dataset and checkpoint
DATASET_NAME=$1

if [[ $DATASET_NAME == "e2h" ]]; then
    SPLIT=train
    MODEL_PATH=/root/code/assets/ckpts/e2h_ema_0.9999_420000_adapted.pt
elif [[ $DATASET_NAME == "diode" ]]; then
    SPLIT=train
    MODEL_PATH=/root/code/assets/ckpts/diode_ema_0.9999_440000.pt
fi

source scripts/args.sh $DATASET_NAME

# Number of function evaluations (NFE)
NFE=$2

# Sampler
GEN_SAMPLER=$3

if [[ $GEN_SAMPLER == "heun" ]]; then
    N=$(echo "$NFE" | awk '{print ($1 + 1) / 3}')
    N=$(printf "%.0f" "$N")
    # Default setting in the DDBM paper
    CHURN_STEP_RATIO=0.33
elif [[ $GEN_SAMPLER == "dbim" ]]; then
    N=$((NFE-1))
    ETA=$4
elif [[ $GEN_SAMPLER == "dbim_high_order" ]]; then
    N=$((NFE-1))
    ORDER=$4
elif [[ $GEN_SAMPLER == "dbmsolver" ]]; then
    N=$(echo "$NFE" | awk '{print ($1 + 2) / 2}')
    SDE_ORDER=$4
elif [[ $GEN_SAMPLER == "ground_truth" ]]; then
    N=1
fi

# torchrun $run_args sample.py --steps $N --sampler $GEN_SAMPLER --batch_size $BS \

torchrun $run_args sample.py --steps $N --sampler $GEN_SAMPLER --batch_size $BS \
 --model_path $MODEL_PATH --class_cond $CLASS_COND --noise_schedule $PRED \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \
 --condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
 --dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \
 --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT\
 ${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} --use_fp16=$USE_FP16 \
 ${ETA:+ --eta="${ETA}"} --sde_order=${SDE_ORDER} \
 ${ORDER:+ --order="${ORDER}"}
