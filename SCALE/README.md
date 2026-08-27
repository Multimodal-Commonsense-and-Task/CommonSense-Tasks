# SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models (ICML'26 Spotlight)

Official implementation of **"SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models"**.

[`Hyeonbeom Choi`](https://hyeonbeomchoi.github.io)\* | [`Daechul Ahn`](https://dcahn12.github.io/)\* | [`Youhan Lee`](https://youhanlee.github.io) | [`Taewook Kang`](https://twkang43.github.io) | [`Seongwon Cho`](https://seongwon980.github.io/) | [`Jonghyun Choi`](https://ppolon.github.io/)†

<sub><span style="color:gray">\* Equal contribution &nbsp;&nbsp; † Corresponding author</span></sub>

[[Paper]](https://arxiv.org/pdf/2602.04208) [[Project Page]](https://dcahn12.github.io/projects/scale/)

## 🧭 Overview

> **TL;DR.** VLA models have emerged as a promising paradigm for general-purpose robotic control, with test-time scaling (TTS) gaining attention to enhance robustness beyond training. However, existing TTS methods require additional training, verifiers, and multiple forward passes, making them impractical for deployment. Moreover, they intervene only at action decoding while keeping visual representations fixed—insufficient under perceptual ambiguity. We propose **SCALE**, a simple inference strategy that jointly modulates visual perception and action based on self-uncertainty, inspired by Active Inference theory—requiring **no additional training, no verifier, and only a single forward pass.**

SCALE conditions inference on the model's own **self-uncertainty** `u`, estimated from the output logits as the
expected log-likelihood ratio between a one-hot (full-certainty) and a uniform (full-ambiguity) reference distribution:

```
u_k = D_KL(p || q_low) − D_KL(p || q_high)        
```

and uses it to jointly modulate two things in a **single forward pass** :

1. **Adaptive Action Decoding** : per-token sampling temperature `τ_k = T₀ · σ(u_k)` — near-greedy when confident, explorative when uncertain.
2. **Adaptive Visual Attention** : vision-encoder attention temperature `γ_t = κ^tanh(Δu_{t-1})` — sharpens focus when confident (`γ < 1`), broadens exploration when uncertain (`γ > 1`).

This repository provides the **OpenVLA on LIBERO** instantiation of SCALE used in the paper.

## 🔊 Updates

- [x] Release the paper on [arXiv](https://arxiv.org/abs/2602.04208)
- [x] Open the [project page](https://dcahn12.github.io/projects/scale/) for SCALE!
- [x] Release the code for SCALE (OpenVLA + LIBERO)

## 📦 Installation

```bash
# 1) Create and activate the environment
conda create -n scale python=3.10 -y
conda activate scale

# 2) Install PyTorch + CUDA toolchain
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 cuda-compiler -c pytorch -c nvidia -y

# 3) Install this package (its deps pin torch==2.2.0 / torchvision==0.17.0 / torchaudio==2.2.0)
pip install -e .

# 4) Install Flash-Attention 2
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# 5) Clone & install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO --config-settings editable_mode=compat

# 6) Install LIBERO's extra runtime requirements, then pin compatible versions
pip install -r experiments/robot/libero/libero_requirements.txt
pip install numpy==1.26.4 mujoco==3.3.2
```

## 🚀 Evaluation

Pick the GPU with `CUDA_VISIBLE_DEVICES`. The simplest entry point:

```bash
# SCALE (paper defaults from configs/scale.yaml).
# The checkpoint is auto-selected from --task_suite (pass --pretrained_checkpoint only to override).
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --task_suite libero_10 \
    --decoding_mode scale
```

### Decoding modes

`--decoding_mode` selects the inference strategy. Supported modes and their flags:

| Mode | Description | Flags (paper baseline config) |
|---|---|---|
| `greedy` | Standard greedy / argmax (OpenVLA default) | — |
| `temp`   | Temperature sampling | `--temperature 1.0` |
| `topk`   | Top-k sampling | `--top_k 40 --temperature 0.7` |
| `topp`   | Top-p (nucleus) sampling | `--top_p 0.9` |
| `scale`  | **SCALE (ours)** | reads `configs/scale.yaml` (override with `--T0`, `--kappa`, ...) |

Examples:

```bash
# Greedy baseline
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --task_suite libero_spatial --decoding_mode greedy

# Top-k baseline (k=40, t=0.7)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --task_suite libero_object --decoding_mode topk --top_k 40 --temperature 0.7
```

Or use the convenience wrapper:

```bash
# bash experiments/robot/libero/run_libero_eval.sh <GPU_ID> <TASK_SUITE> <DECODING_MODE>
bash experiments/robot/libero/run_libero_eval.sh 0 libero_10 scale
```

Per-episode logs are written to `experiments/logs/` and replay videos to `rollouts/` (disable with `--save_video False`).

## ⚙️ SCALE hyperparameters

The SCALE defaults live in [`configs/scale.yaml`](configs/scale.yaml) and follow the paper. 
Names are aligned with the paper and any key can be overridden on the command line, e.g. `--T0 1.0 --kappa 2.0 --alpha 0.8`.

## 📖 Citation

```bibtex
@inproceedings{choi2026scale,
  title={SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models},
  author={Hyeonbeom Choi and Daechul Ahn and Youhan Lee and Taewook Kang and Seongwon Cho and Jonghyun Choi},
  booktitle={ICML},
  year={2026}
}
```

## 🙏 Acknowledgements

This codebase is built on top of [OpenVLA](https://github.com/openvla/openvla) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
