export PYTHONPATH=$PYTHONPATH:./

DATASET_NAME=$1
NFE=$2
GEN_SAMPLER=$3

if [[ $DATASET_NAME == "e2h" ]]; then
    SPLIT=train
    PREFIX=e2h_ema_0.9999_420000_adapted/sample_420000
    REF_PATH=assets/stats/edges2handbags_ref_64_data.npz
    SAMPLE_NAME=samples_138567x64x64x3_nfe${NFE}.npz
elif [[ $DATASET_NAME == "diode" ]]; then
    SPLIT=train
    PREFIX=diode_ema_0.9999_440000_adapted/sample_440000
    REF_PATH=assets/stats/diode_ref_256_data.npz
    SAMPLE_NAME=samples_16502x256x256x3_nfe${NFE}.npz
fi

if [[ $GEN_SAMPLER == "heun" ]]; then
    N=$(echo "$NFE" | awk '{print ($1 + 1) / 3}')
    N=$(printf "%.0f" "$N")
    SAMPLER="heun"
elif [[ $GEN_SAMPLER == "dbim" ]]; then
    N=$((NFE-1))
    ETA=$4
    SAMPLER="dbim_eta=${ETA}"
elif [[ $GEN_SAMPLER == "dbim_high_order" ]]; then
    N=$((NFE-1))
    ORDER=$4
    SAMPLER="dbim_order=${ORDER}"
elif [[ $GEN_SAMPLER == "dbmsolver" ]]; then
    N=$(echo "$NFE" | awk '{print ($1 + 2) / 2}')
    SDE_ORDER=$4
    SAMPLER="dbmsolver_sde_order=${SDE_ORDER}"
elif [[ $GEN_SAMPLER == "ground_truth" ]]; then
    N=1
    SAMPLER="ground_truth"
fi

# For example:
# SAMPLE_PATH="workdir/e2h_ema_0.9999_420000_adapted/sample_420000/split=train/dbim_eta=0.0/steps=4/samples_138567x64x64x3_nfe5.npz"
# SAMPLE_PATH="workdir/diode_ema_0.9999_440000_adapted/sample_440000/split=train/dbim_eta=0.0/steps=4/samples_16502x256x256x3_nfe5.npz"
# SAMPLE_PATH="workdir/imagenet256_inpaint_ema_0.9999_400000/sample_400000/split=test/dbim_order=3/steps=9/samples_10000x256x256x3_nfe10.npz"

SAMPLE_DIR=workdir/${PREFIX}/split=${SPLIT}/${SAMPLER}/steps=${N}
SAMPLE_PATH=${SAMPLE_DIR}/${SAMPLE_NAME}

if [[ $DATASET_NAME == "e2h" || $DATASET_NAME == "diode" ]]; then
    python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric fid
    python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric lpips
elif [[ $DATASET_NAME == "imagenet_inpaint_center" ]]; then
    LABEL_PATH=${SAMPLE_DIR}/${LABEL_NAME}
    python evaluation/compute_metrices_imagenet.py --ckpt $SAMPLE_PATH --label $LABEL_PATH --dataset-dir $DATA_DIR
    python evaluations/evaluator.py "" $SAMPLE_PATH --metric is
fi