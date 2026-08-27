# FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniforge3/bin:${PATH}"
ARG PATH="/root/miniforge3/bin:${PATH}"

RUN apt-get update 
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y gcc libstdc++6 vim git tmux wget htop libgl1 libglx-mesa0 libglib2.0-0 libsm6 libxrender1 libxext6

ENV PYTHONIOENCODING=UTF-8

# Miniforge:
# RUN wget https://github.com/conda-forge/miniforge/releases/download/24.11.0-0/Miniforge3-24.11.0-0-Linux-x86_64.sh
# RUN wget https://github.com/conda-forge/miniforge/releases/download/24.9.0-0/Miniforge3-24.9.0-0-Linux-x86_64.sh
RUN wget https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh

RUN mkdir /root/.conda
RUN bash Miniforge3-25.3.1-0-Linux-x86_64.sh -b
RUN rm -f Miniforge3-25.3.1-0-Linux-x86_64.sh
RUN conda init bash

RUN conda update -y -n base conda
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba

RUN conda install -y python=3.10

RUN python -m pip install packaging ninja

RUN conda update -y --all

RUN conda install -y -c conda-forge mpi4py openmpi wheel==0.43.0

RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install --upgrade setuptools==57.0.0
RUN python3 -m pip install --upgrade setuptools==80.8.0
# RUN pip3 install torch==2.1.0+cu121 torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# RUN python -m pip install torch==2.2.0+cu118 torchaudio==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# RUN python -m pip install torch==2.2.0+cu121 torchaudio==2.2.0+cu121 torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# RUN python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129
RUN python -m pip install jvp_flash_attention==0.10.0

RUN python -m pip install -U albumentationsx==2.0.11
RUN python -m pip install clean-fid
RUN python -m pip install chardet pytorch-msssim lpips munch
RUN python -m pip install torch-summary==1.4.5 torch-fidelity==0.3.0

RUN python -m pip install --force-reinstall charset-normalizer==3.1.0
RUN python -m pip install torchmetrics[image]==1.8.2
# RUN python -m pip install blobfile tqdm numpy==1.25.2 scipy==1.11.1 pandas Cython piq==0.7.0
RUN python -m pip install blobfile
RUN python -m pip install tqdm
# RUN python -m pip install numpy==1.25.2
# RUN python -m pip install pkgconfig 
# RUN apt install -y libopenblas-dev cmake
RUN python -m pip install scipy
RUN python -m pip install Cython
RUN python -m pip install piq==0.7.0
RUN python -m pip install joblib==0.14.0 
RUN python -m pip install lmdb 
# RUN python3 -m pip install --upgrade setuptools
RUN python -m pip install clip@git+https://github.com/openai/CLIP.git pillow
# RUN python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
# RUN python -m pip install flash-attn==2.8.3 --no-build-isolation
RUN python -m pip install nvidia-ml-py3 legacy dill nvidia-ml-py3
RUN python -m pip install timm==1.0.15 wandb

RUN python -m pip install matplotlib
RUN apt update
RUN apt install -y libopenmpi-dev
ARG DEBIAN_FRONTEND=teletype(base)