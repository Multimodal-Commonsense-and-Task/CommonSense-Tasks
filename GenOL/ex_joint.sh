# #!/bin/bash
# #SBATCH -p suma_rtx4090
# #SBATCH -q big_qos
# #SBATCH --gres=gpu:1
# ##SBATCH -c 32

# ulimit -u 200000
# source ~/.bashrc
# ml purge
# conda init bash
# conda activate generate

# --------------------------IMPORTANT-------------------------- #
MODEL_NAME="resnet18" # vit
DATASET="DomainNet" # PACS_final, DomainNet, cifar10, NICO, cct
TYPE="train_ma"
BATCH_SIZE=128
LEARNING_RATE=1e-4
NUM_EPOCHS=10
GPUS=("4")
# --------------------------IMPORTANT-------------------------- #

LOG_FILE="./logs/${DATASET}_${TYPE}_${MODEL_NAME}.log"
CUDA_VISIBLE_DEVICES=${GPUS[0]} nohup python main_joint.py --model_name $MODEL_NAME --dataset $DATASET \
--type $TYPE --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS \
> $LOG_FILE 2>&1 &