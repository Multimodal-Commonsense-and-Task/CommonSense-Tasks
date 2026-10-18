# OASIS: Online Sample Selection for Continual Instruction Tuning

<p align="center">
  <a href="https://aclanthology.org/2026.acl-long.158/">
    <img src="https://img.shields.io/badge/ACL_2026-Paper-blue" alt="Paper">
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-yellow" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-orange" alt="PyTorch">
</p>

<p align="center">
  Official implementation of <br>
  <a href="https://arxiv.org/abs/2506.02011">
    <strong>OASIS: Online Sample Selection for Continual Instruction Tuning</strong>
  </a>
</p>


# Environment Setup

```bash
conda create -n oasis python=3.10 -y
conda activate oasis
# Torch 2.2.1 requires NumPy 1.x.
pip install transformers==4.49.0 numpy==1.26.4
# Keep these CUDA 11.8 packages on their matching versions.
pip install torch==2.2.1 torchvision==0.17.1 xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118
# Prebuilt wheel for Python 3.10, CUDA 11.8, Torch 2.2, and the old C++ ABI.
pip install --no-cache-dir --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7%2Bcu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
pip install peft==0.11.1 trl==0.8.6 deepspeed==0.16.0 bitsandbytes==0.43.3 pandas kornia opencv-python==4.10.0.84 timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf loguru captum==0.7.0 POT jsonlines imageio accelerate==0.29.3 numpy==1.26.4 faiss-cpu h5py 'qpsolvers[osqp]' qwen_vl_utils rich decord scikit-learn==1.5.2 einops ninja 'fsspec[http]==2026.6.0' torch==2.2.1 torchvision==0.17.1 xformers==0.0.25 --extra-index-url https://download.pytorch.org/whl/cu118

# Verify the CUDA extension stack before starting training.
python -c "import torch, numpy, flash_attn, xformers; print(torch.__version__, torch.version.cuda, numpy.__version__, flash_attn.__version__, xformers.__version__)"

```

`xformers==0.0.25` is built specifically for `torch==2.2.1`. Keep the
versions above pinned: newer `bitsandbytes` and `captum` releases require a
newer Torch release and cause pip to replace Torch 2.2.1, making the xFormers
wheel incompatible. Torch 2.2.1 also requires NumPy 1.x. The FlashAttention
wheel is pinned to the same Python, CUDA, Torch, and C++ ABI combination.


-----------------------------------------------------------------

# Quick Start

## Training

```bash
conda activate oasis
bash train_VLM_CL.sh
```

On the first run, the model is downloaded from Hugging Face into the local
cache. Ensure the machine has internet access and sufficient disk space before
starting training.
- `bash train_VLM_CL.sh`
    - Arguments:
        - NOTE: Experiment name **All the models are saved in client_states_$NOTE folder**
        - MODE: Name of method (method in `selection_methods/method_manager.py`)

## Evaluation
```bash
bash eval.sh
```
- `bash eval.sh`
    - Arguments:
        - NOTE: experiment name **Models in client_states_$NOTE folder will be loaded**
        - MODE: Name of method (method in `selection_methods/method_manager.py`)

-----------------------------------------------------------------

## Citation

```bibtex
@article{lee2025oasis,
  title={Oasis: Online sample selection for continual visual instruction tuning},
  author={Lee, Minjae and Seo, Minhyuk and Qu, Tingyu and Tuytelaars, Tinne and Choi, Jonghyun},
  journal={arXiv preprint arXiv:2506.02011},
  year={2025}
}
```

---

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
