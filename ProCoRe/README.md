# The Official PyTorch Implementation of ProCoRe (Proposition-based Contrastive Reinforcement learning)

**Visual Grounding First, Multimodal In-context Learning Follows**<br>
Minhyuk Seo, Minjae Lee, Chaeeun Lee, Wei Lin, Muhammad Jehanzeb Mirza, Tinne Tuytelaars, Jonghyun Choi<br>
NeurIPS 2026 · [arXiv](https://arxiv.org/abs/2610.30592)

## Repository snapshot

This copy keeps the implementation and lightweight configuration files in the CommonSense-Tasks repository. Large data lists and generated MTV artifacts are omitted: `datalists/`, `MTV/storage/`, `MTV/storage_json/`, `MTV/origin_data/`, and `MTV/existing_benchmark_json/`. Retrieve those resources from the [upstream repository](https://github.com/snumprlab/procore) when needed.

## Environment

### Training environment
Make sure `nvidia-smi` and `nvcc` both report CUDA 11.6 or later.

```bash
conda create -n mmicl python=3.10
conda activate mmicl
pip install git+https://github.com/davidbau/baukit@main#egg=baukit

pip install torch==2.2.1 torchvision xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.7 --no-build-isolation

conda install -c conda-forge cudatoolkit-dev -y
conda install conda-forge::openjdk=8
```

Then install the packages for the model family you use.

</details>

<details>
<summary>Qwen3-VL / InternVL3.5 / LLaVA-OneVision-1.5</summary>

```bash
pip install peft==0.11.1 trl==0.8.6 deepspeed==0.16.0 bitsandbytes pandas kornia opencv-python torch_optimizer easydict pycocoevalcap loguru captum POT numpy==1.26.4 faiss-cpu h5py qpsolvers "qpsolvers[osqp]" scikit-learn math_verify evaluate latex2sympy2_extended mpi4py imageio decord tabulate openai "trl[vllm]" sentence-transformers FactualSceneGraph capture-metric spacy openpyxl accelerate==1.9.0 datasets==2.19.2 einops==0.6.0 ftfy==6.1.3 hydra-core==1.3.2 jsonlines==4.0.0 megatron-energon==5.0.0 nltk==3.9.1 omegaconf==2.3.0 pandarallel==1.6.5 pre-commit==3.2.2 protobuf==3.20.2 py-cpuinfo==9.0.0 qwen_vl_utils==0.0.14 sentencepiece==0.1.98 tiktoken==0.5.2 timm==1.0.3 wandb==0.21.0 wrapt==1.16.0 transformers==4.57.3 torch==2.4.0 torchvision==0.19.0 ollama
```
</details>

For GRPO-based training (ProCoRe and the RL baselines), also install:
```bash
pip install trl==0.20.0 open_clip_torch bert-score
```

### API keys
ProCoRe decomposes captions into propositions with GPT by default, and some models are gated on Hugging Face. Set both keys in your shell. Do not hard-code them in the scripts.
```bash
export OPENAI_API_KEY=<your OpenAI key>
export HF_TOKEN=<your Hugging Face token>
```


## Training

```bash
bash train_GRPO_VLM.sh
```
- `REWARD`: the reward function, defined in `utils/reward_functions.py`.
  - `procore`: **ProCoRe**. It decomposes the generated caption pair into propositions and rewards entailment with the matching image while penalizing the mismatched one, plus proposition-image similarity rewards.
- `MODEL_ARCH`: `qwen3-8b`, `internvl3_5-8b`, `onevision1_5-8b`
- `DATASET_NAME`: the contrastive training data. The default is `inpaintcoco`.
- `--proposition-model`: the model that decomposes captions into propositions. The default is `gpt`.
- `GPU`: the GPU index. `MASTER_PORT`: the DeepSpeed port.

Run `bash train_GRPO_VLM.sh --help` to see all reward-specific options.

```bash
bash train_VLM.sh
```
- `METHOD`: `dara`, `mimic_mixed`, `mt-icl`, `Finetune`, `stv`
- `MODEL_ARCH`: `qwen3-8b`, `internvl3_5-8b`, `onevision1_5-8b`
- `DATASET_NAME`: the ICL training data, e.g. `truemicl`, `mic`, `vliclv2`, or `isekai`

## Evaluation
```bash
bash eval.sh
```
- `METHOD`: `grpo` (ProCoRe and the RL baselines), `Finetune`, `mt-icl`, `dara`, `mimic`, `fv`, `mtv`, or `None` (the off-the-shelf model)
- `REWARD` / `EVAL_REWARD`: for `METHOD=grpo`, the reward used in training. It selects the checkpoint.
- `MODEL_ARCHS`: `qwen3-8b`, `internvl3_5-8b`, `onevision1_5-8b`, or `idefics-8b`
- `DATASET_NAME`: the evaluation benchmark, e.g. `truemicl`, `mic`, `vliclv2`, `isekai`, `pororov4`, `storystreamv2`, `flintstonesv3`, or `SMMILE`
- `ORDERS`: the demonstration ordering: `random`, `multimodal`, or `mmices`
- `EVAL_SHOTS`: `0-shot`, `1-shot`, `2-shot`, `4-shot`, `8-shot`, ..., or `mixed`

The training hyperparameters (e.g. `LR`, `EPOCH`, and `RND_SEED`) must match the ones used in training, so that `eval.sh` loads the right checkpoint.

## Citation
```bibtex
@inproceedings{seo2026visual,
  title={Visual Grounding First, Multimodal In-context Learning Follows},
  author={Seo, Minhyuk and Lee, Minjae and Lee, Chaeeun and Lin, Wei and Mirza, Muhammad Jehanzeb and Tuytelaars, Tinne and Choi, Jonghyun},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  url={https://arxiv.org/abs/2610.30592}
}
```

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
