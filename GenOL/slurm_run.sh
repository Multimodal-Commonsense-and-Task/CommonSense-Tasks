#!/bin/bash
##SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH -c 64

source ~/.bashrc
conda init bash
conda activate generate

./new_requirements.sh