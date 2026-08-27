# Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients

<p align="center">
  <a href="https://openreview.net/forum?id=0g5Dk4Qfh0">
    <img src="https://img.shields.io/badge/ICLR_2026-Paper-blue" alt="Paper">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-yellow" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-orange" alt="PyTorch">
</p>

<p align="center">
  Official implementation of <br>
  <a href="https://openreview.net/forum?id=0g5Dk4Qfh0">
    <strong>Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients</strong>
  </a>
</p>

---

## Overview

This repository contains the official code for **Co-LoRA (ICLR 2026)**.  
It supports training and evaluation of **FedMosaic**, our federated continual learning framework for **heterogeneous multi-modal clients**, together with several federated learning baselines.

The central idea is **Co-LoRA**, a dimension-agnostic LoRA design that enables collaborative knowledge sharing across heterogeneous client models.

### Main features

- Federated continual learning for heterogeneous vision-language clients
- Support for multiple baseline methods alongside **FedMosaic**
- Scenario-based client/model/task assignment through JSON configuration
- Training and evaluation pipelines for DRAKE, HFLB, and related benchmarks

---

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Setup](#dataset-setup)
- [Federated Learning Configuration](#federated-learning-configuration)
- [Training](#training)
  - [Quick Start](#quick-start)
  - [Script Arguments](#script-arguments)
  - [Method-Specific Settings](#method-specific-settings)
  - [Dataset-Specific Settings](#dataset-specific-settings)
- [Evaluation](#evaluation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Citation](#citation)

---

## Environment Setup

We recommend using a fresh conda environment.

```bash
conda create -n fcl2 python=3.10
conda activate fcl2

pip install transformers==4.47.1
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.8 --no-build-isolation
pip install peft==0.14.0 bitsandbytes pandas kornia opencv-python timm \
    torch_optimizer easydict pycocoevalcap sentencepiece protobuf \
    trl==0.8.6 deepspeed==0.15.2 loguru captum POT jsonlines \
    numpy==1.26.4 accelerate==0.29.3 nevergrad
pip install -U scikit-learn
```

---

## Dataset Setup

Place all datasets under the `dataset/` folder at the project root.

DRAKE benchmark is available in Hugging Face:
- [SNUMPR/DRAKE](https://huggingface.co/datasets/SNUMPR/DRAKE): large-scale multi-modal personalized federated continual learning benchmark
- [SNUMPR/HFLB](https://huggingface.co/datasets/SNUMPR/HFLB): 


#### Download Instruction:
To download all at once:
```bash
mkdir dataset
cd dataset
huggingface-cli download SNUMPR/DRAKE --local-dir ./ --repo-type dataset
```
But we **highly recommend** to download each dataset in DRAKE separately due to enormous dataset size, by:
```bash
# example: downloading Fashion200K dataset
cd dataset
huggingface-cli download SNUMPR/DRAKE Fashion200K.tar --local-dir ./ --repo-type dataset
tar -xvf Fashion200K.tar
rm Fashion200K.tar
```

---

<details>
<summary>Additionally download LLaVA-v1.5 fine-tuning dataset</summary>
Additionally download LLaVA-v1.5 fine-tuning dataset to use it as public dataset

```bash
# Inside dataset/ folder,

mkdir llava_finetune
cd llava_finetune

# original llava json file
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json?download=true
mv llava_v1_5_mix665k.json?download=true llava_v1_5_mix665k.json

# coco
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
cd..

# gqa
mkdir gqa
cd gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
rm images.zip
cd ..

# textvqa
mkdir textvqa
cd textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
rm train_val_images.zip
cd ..

# vg
mkdir vg
cd vg
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip images.zip
rm images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images2.zip
rm images2.zip
cd ..

# ocr_vqa
huggingface-cli download SNUMPR/DRAKE llava_ft/ocr_vqa.tar --local-dir ./ --repo-type dataset
tar -xvf ocr_vqa.tar
rm ocr_vqa.tar

# splited json files
huggingface-cli download SNUMPR/DRAKE llava_ft/llava_ft_jsons.tar --local-dir ./ --repo-type dataset
tar -xvf llava_ft_jsons.tar
rm llava_ft_jsons.tar

```
</details>

#### Dataset Structure

After extraction, the directory layout should look like:

```
dataset/
├── Fashion200K/
│   ├── full/images/             # raw image files
│   ├── train/
│   │   ├── dataset-0.json       # task split 0
│   │   ├── dataset-1.json       # task split 1
│   │   └── ...
│   └── test/
│       ├── dataset-0.json
│       ├── dataset-1.json
│       └── ...
└── <other_datasets>/
    ├── <image_folder>/
    ├── train/
    │   ├── dataset-0.json
    │   └── ...
    └── test/
        ├── dataset-0.json
        └── ...
```
Each `dataset-<id>.json` corresponds to a task subset defined by our benchmark split. 

Please refer to the paper and [scenarios/DRAKE_task_list.json](scenarios/DRAKE_task_list.json).

## Federated Learning Configuration

Client tasks and model assignments are managed via a single JSON scenario file. For a summary of all scenarios used in the paper, see [docs/SCENARIO.md](docs/SCENARIO.md).

---

## Training

### Quick Start

```bash
bash train_VLM_CL.sh
```

### Script Arguments

Before running, edit the top section of `train_VLM_CL.sh` to configure your experiment. The key variables are:

#### General Settings

| Variable | Default | Description |
|---|---|---|
| `NOTE` | `"debug_fedmosaic"` | Experiment name used for logging and checkpoint saving |
| `MODE` | `"fedmosaic"` | Federated learning method (see [Method-Specific Settings](#method-specific-settings)) |

#### Federated Learning Settings

| Variable | Default | Description |
|---|---|---|
| `SCENARIO` | `DRAKE_hetero_llava_llama_1B_3B` | Scenario name defined in the `scenarios/` folder |
| `NUM_ROUNDS` | `5` | Number of federated communication rounds per task |
| `NUM_TASKS` | `4` | Number of tasks per client defined in the scenario JSON file |
| `NUM_CLIENTS` | `10` | Number of federated clients defined in the scenario JSOn file |
| `NUM_ITER` | `100` | Local training iterations per round (method-dependent, see below) |
| `IS_MULTIMODAL` | `True` | Enable multimodal (vision-language) training |
| `IS_CONTINUAL` | `True` | Enable PFL-Dynamic setup |
| `--is_cross_model_series` | `False` | set `True` if FL scenario contains different model series, e.g., llama and qwen |
#### Optimization Settings

| Variable | Default | Description |
|---|---|---|
| `BATCHSIZE` | `4` | Gradient accumulation steps (effective total batch size) |
| `LR` | `2e-5` | Base learning rate for lora weights (A, B) |
| `MM_PROJECTOR_LR` | `5e-5` | Learning rate for other weights (e.g., Co-LoRA's P & Q) |
| `SCHED_NAME` | `"constant"` | LR scheduler type (`constant` or `cosine`) |

---

### Method-Specific Settings

Different federated methods require specific `NUM_ITER` and `BATCHSIZE` settings for fair computational comparison (See Sec.A.8). Set `MODE` to one of the following and adjust accordingly:

| `MODE` | `NUM_ITER` | `BATCHSIZE` | Notes |
|---|---|---|---|
| `fedmosaic` | `94` | `4` | Set `USE_TASK_VECTOR=True` |
| `sft` / `fedavg` / `feddpa` | `100` | `4` | |
| `fedsim` / `takfl` | `75` | `4` | |
| `fedmkt` | `60` | `4` | |
| `ditto` / `perada` | `50` | `8` | |
| `feddat` | `43` | `8` | |

> **Note**
> - `USE_TASK_VECTOR=True` should **only** be set for `fedmosaic`.
> - Methods that update both local and global modules during local training may require a larger batch size under `deepspeed` to preserve fair effective updates across methods.

---

### Dataset-Specific Settings

Adjust learning rates and multimodal flags depending on the dataset used:

| Dataset | `LR` | `MM_PROJECTOR_LR` | `IS_MULTIMODAL` | `--lora_r` | `--lora_alpha` | `SCHED_NAME` | `sft NUM_ITER` |
|---|---|---|---|---|---|---|---|
| DRAKE / HFLB | `2e-5` | `5e-5` | `True` | `128` | `256` | `constant` | `100` |
| Fed-Scope / Fed-aya | `3e-4` | `5e-4` | `False` | `16` | `32` | `cosine` | `30` / `50` |
| Fed-LLM-Large | `1e-4` | `5e-4` | `False` | `16` | `32` | `cosine` | `10`? |

---

## Evaluation

Default **Self** and **Others** evaluation is performed using:

```bash
bash eval_scripts/eval.sh
```

> Make sure `MODE`, `NOTE` and `SCENARIO` in `eval_scripts/eval.sh` match the values used during training so that the correct checkpoints are loaded.

### Script Arguments

| Variable | Default | Description |
|---|---|---|
| `--zeroshot` | `False` | Enable base model evaluation without loading trained checkpoint |
| `--eval_all` | `False` | Enable evaluation on all tasks in the given scenarios, including `Self` tasks and `Others` tasks |


> If set `--eval_all True`, we highly recommend to run the evaluation per client in parallel, using arguments `--eval_client_start`, `--eval_client_end`, `--eval_client_eval_start`, `--eval_client_eval_end`.

---

## Pretrained Checkpoints

For Co-LoRA, AB-aligned checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1VcqZMGiVuyS59AnuTJ_q5Ky6nVmDPWFZ?usp=drive_link).

To reproduce them from scratch:

```bash
bash train_abalign.sh
```

---

## 📄 Citation

If you find our work helpful in your research, please consider citing our paper. We'd really appreciate it! 🙏

```bibtex
@inproceedings{seo2026colora,
  title     = {Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients},
  author={Seo, Minhyuk and Kim, Taeheon and Lee, Hankook and Choi, Jonghyun and Tuytelaars, Tinne},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=0g5Dk4Qfh0}
}
```

---

## 🙌 Acknowledgements

We sincerely thank the open-source community — this work builds on top of many excellent projects including [LLaVA](https://github.com/haotian-liu/LLaVA), [HuggingFace Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [DeepSpeed](https://github.com/microsoft/DeepSpeed). We hope this codebase serves as a useful starting point for the federated and continual learning community, and we warmly welcome any questions, issues, or contributions. 😊

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
