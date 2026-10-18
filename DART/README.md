# Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts (ECCV 2026)

Official implementation of **"Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts"** (ECCV 2026).

[`Taewook Kang`](https://twkang43.github.io)\* | [`Taeheon Kim`](https://ta3h30nk1m.github.io/)\* | [`Donghyun Shin`](https://github.com/Donghyun1228) | [`Jonghyun Choi`](https://ppolon.github.io/)†

<sub><span style="color:gray">\* Equal contribution &nbsp;&nbsp; † Corresponding author</span></sub>

[[Paper]](https://arxiv.org/abs/2607.00666) [[Project Page]](https://twkang43.github.io/projects/dart/) [[Checkpoints & Datasets]](https://huggingface.co/collections/SNUMPR/dart)


## 🧭 Overview

> **TL;DR.** **Domain ARiThmetic (DART)** adapts multi-task vision-language-action (VLA) models to environmental shifts using only **one demonstration of one task** through subspace-aligned weight arithmetic.

This repository provides:

- one-shot fine-tuning scripts for `π₀.₅` on LIBERO visual-shift data,
- the DART implementation in [`domain_arithmetic/`](domain_arithmetic/),
- `LIBERO-view` evaluation scripts.

## 📢 Updates

- [x] Release the paper on [arXiv](https://arxiv.org/abs/2607.00666).
- [x] Launch the [project page](https://twkang43.github.io/projects/dart/).
- [x] Release code.
- [x] Upload one-shot fine-tuning datasets and checkpoints.

## 📦 Installation

Clone the repository with submodules:

```bash
git clone --recurse-submodules git@github.com:snumprlab/dart.git
cd dart
```

If you already cloned the repository, initialize the submodules manually:

```bash
git submodule update --init --recursive
```

We use [`uv`](https://docs.astral.sh/uv/) to manage Python dependencies. After installing `uv`, set up the environment with:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is required because LeRobot is pulled as a dependency.


## 💾 Fine-tune VLA Models with One-Shot Data

### Base VLA Model

We use the `pi05_libero` checkpoint from [openpi](https://github.com/Physical-Intelligence/openpi) as the source-trained multi-task VLA model. The checkpoint is trained on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark:

```text
gs://openpi-assets/checkpoints/pi05_libero
```

By default, OpenPI downloads this checkpoint from `gs://openpi-assets` and caches it under `~/.cache/openpi`. To change the cache location, set:

```bash
export OPENPI_DATA_HOME=/path/to/openpi_cache
```

### Prepare the Dataset

One-shot fine-tuning expects a LeRobot-format LIBERO dataset. Set `DATA_REPO_ID` to either a local dataset path or a Hugging Face dataset repo id.

Supported third-person camera views are:

```text
original (source), small, medium, large
```

For visual perturbations, use a separate dataset for each perturbation type.

### Run One-Shot Fine-Tuning

Use [`exec/finetune.sh`](exec/finetune.sh) to fine-tune `π₀.₅` on a selected camera view:

```bash
DATA_REPO_ID=/path/to/dataset \
CHECKPOINT_BASE_DIR=/path/to/checkpoints \
CAMERA_VIEW=medium \
CUDA_VISIBLE_DEVICES={GPU IDs} \
bash exec/finetune.sh
```

Common options:

| Variable | Default | Description |
| --- | --- | --- |
| `CONFIG` | `pi05_libero_oneshotft` | Training config |
| `CAMERA_VIEW` | `original` | Viewpoint shifts |
| `BATCH_SIZE` | `64` | Training batch size |
| `NUM_TRAIN_STEPS` | `1000` | Number of fine-tuning steps |
| `SAVE_INTERVAL` | `1000` | Checkpoint save interval |


## 🎯 DART: One-Shot VLA Adaptation Through Domain Arithmetic

> ⚠️ Warning: The current DART implementation can require more than 100 GB of RAM. This may be reduced with further optimization.

DART requires three policies:

- `base_policy`: the original source-trained VLA model,
- `policy_src`: a one-shot policy fine-tuned in the source environment,
- `policy_tgt`: a one-shot policy fine-tuned in the target shifted environment.

The adapted policy is computed by applying the target-domain update after subtracting the source-specific component in an aligned subspace.

Run the example script:

```bash
bash exec/run_dart.sh
```

Or run DART directly:

```bash
uv run -m domain_arithmetic.dart \
  --cfg.base_policy.dir /path/to/base_policy \
  --cfg.base_policy.config pi05_libero \
  --cfg.policy_src.dir /path/to/source_oneshot_policy \
  --cfg.policy_src.config pi05_libero_oneshotft \
  --cfg.policy_tgt.dir /path/to/target_oneshot_policy \
  --cfg.policy_tgt.config pi05_libero_oneshotft \
  --cfg.scaling_coef 0.8 \
  --cfg.output_dir /path/to/output/DART_0.8 \
  --cfg.overwrite
```

The merged checkpoint is saved to `--cfg.output_dir` and can be evaluated with the same `pi05_libero` policy config.


## 🚀 Evaluation

Evaluation uses Docker to serve the policy and run [`LIBERO-view`](https://github.com/Donghyun1228/LIBERO_view), a modified version of LIBERO for visual shifts.

Evaluate the default `pi05_libero` checkpoint:

```bash
GPU_ID=0 \
TASK_SUITE_NAME=libero_spatial \
CAMERA_VIEW=medium \
bash exec/eval.sh
```

Evaluate a DART checkpoint:

```bash
GPU_ID=0 \
POLICY_CONFIG=pi05_libero \
CHECKPOINT_DIR=/path/to/output/DART \
TASK_SUITE_NAME=libero_spatial \
CAMERA_VIEW=medium \
bash exec/eval.sh
```

Add `PERTURB` argument to evaluate with visual perturbations:

```bash
GPU_ID=0 \
POLICY_CONFIG=pi05_libero \
CHECKPOINT_DIR=/path/to/output/DART \
TASK_SUITE_NAME=libero_spatial \
CAMERA_VIEW=medium \
PERTURB=light_noise \
bash exec/eval.sh
```

Useful evaluation variables:

| Variable | Default | Description |
| --- | --- | --- |
| `GPU_ID` | `0` | Host GPU id used for evaluation |
| `POLICY_CONFIG` | `pi05_libero` | OpenPI policy config |
| `CHECKPOINT_DIR` | empty | Custom checkpoint path; empty uses the default LIBERO checkpoint |
| `TASK_SUITE_NAME` | `libero_spatial` | LIBERO task suite |
| `CAMERA_VIEW` | `original` | Evaluation camera view |
| `PERTURB` | empty | Optional visual perturbation: `light` or `light_noise` |


## 📖 Citation

```bibtex
@inproceedings{kang2026dart,
  title     = {Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts},
  author    = {Kang, Taewook and Kim, Taeheon and Shin, Donghyun and Choi, Jonghyun},
  booktitle = {ECCV},
  year      = {2026},
}
```


## 🙏 Acknowledgements

This codebase is built on top of [openpi](https://github.com/Physical-Intelligence/openpi) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
