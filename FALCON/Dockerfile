FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniforge3/bin:${PATH}"
ARG PATH="/root/miniforge/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get upgrade -y

RUN apt-get install -y gcc
RUN apt-get install libstdc++6
RUN apt-get upgrade -y

RUN apt-get install -y vim git
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install -y tmux
RUN apt-get install -y wget
RUN apt-get install -y htop

ENV PYTHONIOENCODING=UTF-8

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

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools==80.8.0
RUN python -m pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129

RUN conda install -y -c anaconda numpy h5py blosc-hdf5-plugin prettytable
RUN python3 -m pip install tqdm einops matplotlib scipy protobuf tensorboard opencv-python snntorch==0.5.3 spikingjelly protobuf hdf5plugin yacs
RUN conda install -y -c anaconda numba
ARG DEBIAN_FRONTEND=teletype
