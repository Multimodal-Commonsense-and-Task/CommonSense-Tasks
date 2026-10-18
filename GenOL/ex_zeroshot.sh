# CIL CONFIG
# NOTE="imagenet_sdp_sigma0_mem_10000_iter_0.125"
NOTE="rebuttal4_resnet50_clip_zeroshot_cct_iter2_mem200"
MODE="zs_clip"

K_COEFF="4"
TEMPERATURE="0.125"

TRANSFORM_ON_GPU="--transform_on_gpu"
N_WORKER=4
FUTURE_STEPS=8
EVAL_N_WORKER=4
EVAL_BATCH_SIZE=1000
#USE_KORNIA="--use_kornia"
USE_KORNIA=""
UNFREEZE_RATE=0.5
SEEDS="1"
DATA_DIR=""

GPUS=("7" "1" "2" "3" "4")
DATASET="cct" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=0
REPEAT=1
INIT_CLS=100
USE_AMP=""

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=1000
    TYPES=("web_newsample_rmd_temp_1" "web_newsample_rmd_temp_2" "web_newsample_rmd_equalweight" "web_flickr") #("web_10" "ma" "generated" "web" "web_10")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=6000 FEAT_DIM=14 FEAT_MEM_SIZE=24000
    SAMPLES_PER_TASK=20000 
    ONLINE_ITER=2
    EVAL_POINT="2000 4000 6000 8000 10000"

elif [ "$DATASET" == "aircraft" ]; then
    MEM_SIZE=800
    TYPES=("sdxl_diversified" "web" "train_ma")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip_resnet" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=3
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="1335 2669 3997 5332 6667"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="1331 2665 3998 5334 6667"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="1335 2665 4001 5334 6667"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="1331 2666 4001 5331 6667"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="1332 2665 3999 5330 6667"
    fi

elif [ "$DATASET" == "food101" ]; then
    MEM_SIZE=2000
    TYPES=("train_ma" "sdxl_diversified_newprompt" "web")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip_resnet" VAL_PERIOD=500 EVAL_PERIOD=500
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=3
    EVAL_POINT="3000 6000 9000 12000 15150"

elif [ "$DATASET" == "cct" ]; then
    MEM_SIZE=400
    TYPES=("train_ma") #"train_ma" "sdxl_diversified" "web" "generated" "sdxl_diversified_nofiltering" "RMD_classwise_temp_0_5" "RMD_classwise_temp_1" "RMD_classwise_temp_3"
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip_resnet" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=2
    EVAL_POINT="600 1200 1800 2400"

elif [ "$DATASET" == "PACS" ]; then
    MEM_SIZE=200
    TYPES=("sdxl_diversified")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip_resnet" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=2
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="668 1282 1670"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="648 1036 1670"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="573 1039 1670"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="557 1269 1670"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="570 1282 1670"
    fi

elif [ "$DATASET" == "PACS_final" ]; then
    MEM_SIZE=200
    TYPES=("train_ma") # web_RMD_temp_0_5_W web_inverse_temp_0_5_WF "web_RMD_temp_3" "inverseprob" "bottomk" "topk") #"sampling_4" "sampling_2" "sampling_0_5" "sampling_0_25" "sampling_0_125" #"equalweighted" "ensembled_RMD_0_5_modelwise" "ensembled_RMD_1_modelwise" "ensembled_RMD_3_modelwise" "ensembled_RMD_classwise_temp_0_5" "ensembled_RMD_classwise_temp_1" "PACS_final_ensembled_RMD_classwise_temp_3") #("ensembled_samplewise_RMD_0_5" "ensembled_samplewise_RMD_1" "ensembled_samplewise_RMD_3" "ensembled_samplewise_RMD_5") #("ensembled_RMD_temp1" "ensembled_RMD_temp2" "ensembled_RMD_temp5" "ensembled_RMD_temp10")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip_resnet" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=1002 FEAT_DIM=14 FEAT_MEM_SIZE=4800
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=2

    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="458 764 1333"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="638 948 1333"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="520 1013 1333"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="641 950 1333"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="455 764 1333"
    fi

    # 0_5
    # if [ "$SEEDS" == "1" ]; then
    #     EVAL_POINT="228 380 664"
    # elif [ "$SEEDS" == "2" ]; then
    #     EVAL_POINT="318 472 664"
    # elif [ "$SEEDS" == "3" ]; then
    #     EVAL_POINT="259 505 664"
    # elif [ "$SEEDS" == "4" ]; then
    #     EVAL_POINT="319 473 664"
    # elif [ "$SEEDS" == "5" ]; then
    #     EVAL_POINT="226 380 664"
    # fi

    # 0_25
    # if [ "$SEEDS" == "1" ]; then
    #     EVAL_POINT="113 189 331"
    # elif [ "$SEEDS" == "2" ]; then
    #     EVAL_POINT="159 235 331"
    # elif [ "$SEEDS" == "3" ]; then
    #     EVAL_POINT="129 252 331"
    # elif [ "$SEEDS" == "4" ]; then
    #     EVAL_POINT="159 236 331"
    # elif [ "$SEEDS" == "5" ]; then
    #     EVAL_POINT="112 189 331"
    # fi

    # 0_125
    # if [ "$SEEDS" == "1" ]; then
    #     EVAL_POINT="55 93 164"
    # elif [ "$SEEDS" == "2" ]; then
    #     EVAL_POINT="79 116 164"
    # elif [ "$SEEDS" == "3" ]; then
    #     EVAL_POINT="64 125 164"
    # elif [ "$SEEDS" == "4" ]; then
    #     EVAL_POINT="79 117 164"
    # elif [ "$SEEDS" == "5" ]; then
    #     EVAL_POINT="55 93 164"
    # fi


elif [ "$DATASET" == "OfficeHome" ]; then
    MEM_SIZE=500
    TYPES=("train_ma" "generated" "web")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip_resnet" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=2614 FEAT_DIM=14 FEAT_MEM_SIZE=9600
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=2
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="715 1367 2003 2693 3459"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="714 1395 2068 2744 3459"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="634 1356 2075 2772 3459"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="727 1395 2105 2842 3459"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="621 1317 2002 2740 3459"
    fi

elif [ "$DATASET" == "DomainNet" ]; then
    MEM_SIZE=8000
    TYPES=("newsample_equalweight")
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="clip_resnet" VAL_PERIOD=500 EVAL_PERIOD=1000
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    BASEINIT_SAMPLES=30523 FEAT_DIM=14 FEAT_MEM_SIZE=168000
    SAMPLES_PER_TASK=2000
    ONLINE_ITER=3
    if [ "$SEEDS" == "1" ]; then
        EVAL_POINT="10037 20073 30338 40393 50273"
    elif [ "$SEEDS" == "2" ]; then
        EVAL_POINT="10189 20304 30413 40177 50273"
    elif [ "$SEEDS" == "3" ]; then
        EVAL_POINT="9825 19943 30186 40109 50273"
    elif [ "$SEEDS" == "4" ]; then
        EVAL_POINT="10276 20398 30423 40359 50273"
    elif [ "$SEEDS" == "5" ]; then
        EVAL_POINT="9918 20137 30290 40184 50273"
    fi

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    for index in "${!TYPES[@]}"
    do
        CUDA_VISIBLE_DEVICES=${GPUS[$index]} nohup python zero_shot_main.py --mode $MODE $DATA_DIR \
        --dataset $DATASET --unfreeze_rate $UNFREEZE_RATE $USE_KORNIA --k_coeff $K_COEFF --temperature $TEMPERATURE \
        --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --samples_per_task $SAMPLES_PER_TASK \
        --rnd_seed $RND_SEED --val_memory_size $VAL_SIZE --type_name "${TYPES[$index]}" \
        --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
        --lr $LR --batchsize $BATCHSIZE --mir_cands $MIR_CANDS --eval_point "${EVAL_POINT}" \
        --memory_size $MEM_SIZE $TRANSFORM_ON_GPU --online_iter $ONLINE_ITER \
        --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE \
        --baseinit_samples $BASEINIT_SAMPLES --spatial_feat_dim $FEAT_DIM --feat_memsize $FEAT_MEM_SIZE &
    done
done
