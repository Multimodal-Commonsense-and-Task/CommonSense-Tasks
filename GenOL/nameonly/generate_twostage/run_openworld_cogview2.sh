#!/bin/bash
# 10개에 2시간 40분 -> 160분 -> 20개는 320분, 6시간 충분, 40개는 640분, 13시간 충분
# 25개는 

# Floyd: 1분에 2개정도 -> DomainNet 180개 -> 1시간 반, 5개 class 7시간 반
# Cogview2: 72분에 class 1개, 1분에 3.57개, initialize는 7분정도 걸림.
# 1개에 0.2784분 -> 1개 class에 30분 정도 걸린다고 보면 된다. 40개 class정도 생성하면 될듯.

# 4090
# Floyd: 2.2857/min, 0.4375min/1 image ? 2/min??
# SDXL: 11/min, 0.0909min/1 image
# sdturbo: fast
# cogview2: 4/min, 0.25min/1 image

#SBATCH -p suma_a6000
#SBATCH --gres=gpu:1
source ~/.bashrc
ml purge

conda init bash
conda activate cogview

cd Image-Local-Attention
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python setup.py install
cd ../
CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 python generate_images_openworld.py -m cogview2 -r ./generated_datasets/openworld_diversified_cogview2 -s 539 -e 609
