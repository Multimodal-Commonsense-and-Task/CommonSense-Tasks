DATASET_NAME=$1

if [[ $DATASET_NAME == "e2h" ]]; then
    DATA_DIR=/root/data/edges2handbags
    DATASET=edges2handbags
    IMG_SIZE=64

    NUM_CH=192
    NUM_RES_BLOCKS=3
    ATTN_TYPE=True

    EXP="e2h${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False

    PRED="vp"
elif [[ $DATASET_NAME == "diode" ]]; then
    DATA_DIR=/root/data/DIODE-256
    DATASET=diode
    IMG_SIZE=256

    NUM_CH=256
    NUM_RES_BLOCKS=2
    ATTN_TYPE=True

    EXP="diode${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=16
    DROPOUT=0.1
    CLASS_COND=False

    PRED="vp"
fi
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
    SIGMA_MAX=80.0
    SIGMA_MIN=0.002
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi