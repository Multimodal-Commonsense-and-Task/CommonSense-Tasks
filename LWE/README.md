
# Becoming Experienced Judges: Selective Test-Time Learning for Evaluators (EACL 2026)

Official Implementation of the paper "[Becoming Experienced Judges: Selective Test-Time Learning for Evaluators](https://aclanthology.org/2026.eacl-short.50/)" ([EACL 2026](https://2026.eacl.org/) [Short-**Oral**], [ACL 2026 Workshop on Evaluating Evaluations (EvalEval)](https://evalevalai.com/events/2026-acl-workshop/) [**Oral**]).

<img width="2500" height="1280" alt="lwe_explainer0" src="https://github.com/user-attachments/assets/5c6f7d1f-3457-40f8-8fcf-bdf56dc528b6" />

| Method | Description | Module |
|--------|-------------|--------|
| **Vanilla** | Fixed evaluation prompt. | `methods/vanilla.py` |
| **Sample-Specific Prompt (SSP)** | Static meta-prompt for all samples. Per-sample generated rubric, then judge. | `methods/ssp.py` |
| 🌟 **LWE** | Evolving meta-prompt and per-sample generated rubric. | `methods/lwe.py` |
| 🌟🌟 **Selective LWE** | Vanilla consistency check first -> LWE pipeline only on inconsistent samples. | `methods/selective_lwe.py` |

Prompts are in the `prompts/` directory.



## Setup

```bash
cd lwe
pip install -r requirements.txt
```

Set API keys (add to `.env` or export directly):

```bash
export OPENAI_API_KEY=...       # for GPT models
export GEMINI_API_KEY=...       # for Gemini models
export ANTHROPIC_API_KEY=...    # for Claude models
```



## Data Preparation
<details>
<summary>Click to expand</summary>

### Download [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench)

```bash
python data/scripts/download_vlrewardbench.py --output_dir data/vlrewardbench
```

Saves `data/vlrewardbench/vlrewardbench_test.jsonl` and images under `data/vlrewardbench/images/`.

### Download [Multimodal RewardBench](https://github.com/facebookresearch/multimodal_rewardbench)

```bash
python data/scripts/download_mmrewardbench.py --output_dir data/mmrewardbench
```

Creates `data/mmrewardbench/mmrewardbench_test.jsonl`, containing a random sample of 1000 examples (excluding Hateful Memes), and saves images to `data/mmrewardbench/images/`.

Options:
```bash
# Reproduce the exact 1000 examples used in the paper (recommended)
python data/scripts/download_mmrewardbench.py --paper_ids data/scripts/mmrewardbench_paper_ids.json

# Keep all examples (no subsampling)
python data/scripts/download_mmrewardbench.py --no_subsample

# Custom sample size
python data/scripts/download_mmrewardbench.py --sample_size 500
```

### Data format

Each row in the JSONL must contain:

| Field     | Type         | Description                                                   |
|-----------|--------------|---------------------------------------------------------------|
| `ID`      | str          | Unique sample identifier                                      |
| `Text`    | str          | Question / instruction                                       |
| `Output1` | str          | First candidate response                                     |
| `Output2` | str          | Second candidate response                                    |
| `Better`  | str          | `"Output1"` or `"Output2"` — the ground-truth preferred response |
| `Image`   | str \| null  | Path to image file, or null for text-only                    |

You can run with your own custom data by providing a JSONL file with the above format.

Simply point `dataset.data_path` in your config YAML to your custom JSONL file.

</details>



## Run

Run from the `lwe/` directory.

### GPT

```bash
python judge.py --config configs/gpt_vanilla_vl.yaml # method: vanilla, dataset: VL-Rewardbench
python judge.py --config configs/gpt_vanilla_mm.yaml # method: vanilla, dataset: Multimodal Rewardbench
python judge.py --config configs/gpt_ssp_vl.yaml
python judge.py --config configs/gpt_lwe_vl.yaml
python judge.py --config configs/gpt_selective_lwe_vl.yaml
```

### Gemini

```bash
python judge.py --config configs/gemini_selective_lwe.yaml
```

### Claude

```bash
python judge.py --config configs/claude_selective_lwe.yaml
```

Additionally, you can override the `method`, `seed`, or `model` specified in your YAML config directly from the command line using arguments:

```bash
python judge.py --config configs/gemini_lwe.yaml --method selective_lwe --seed 42 --model gemini-2.5-flash
```

## Outputs

Results are written under `out_dir` (default `runs/`), one subdirectory per run:

```
runs/<method>_<model>_<timestamp>/
  config.yaml                  — copy of the run config
  <dataset>.jsonl              — per-sample results
  cumulative_metrics.jsonl     — accuracy / consistency after each batch
  meta_prompt_v0_initial.txt   — (LWE / Selective-LWE) initial meta-prompt
  meta_prompt_final.txt        — (LWE / Selective-LWE) final meta-prompt
  meta_prompt_snapshots/       — (LWE / Selective-LWE) per-batch snapshots
```

Per-sample fields include `pred`, `acc`, `swap_pred`, `swap_acc`, `consistency`, `pair_acc` where applicable.




## Code Structure
<details>
<summary>Click to expand</summary>

```
lwe/
  judge.py                     # entrypoint
  configs/                     # example YAMLs (vanilla/ssp/lwe/selective_lwe × gpt/gemini/claude)
  data/
    scripts/
      download_vlrewardbench.py  # HuggingFace download: MMInstruction/VL-RewardBench
      download_mmrewardbench.py  # HuggingFace download: syhuggingface/multimodal_rewardbench
    vlrewardbench/             # downloaded data
    mmrewardbench/             # downloaded data
  models/
    base.py                    # BaseModel interface
    gpt.py                     # OpenAI GPT
    gemini.py                  # Google Gemini
    claude.py                  # Anthropic Claude
  methods/
    vanilla.py
    ssp.py
    lwe.py
    selective_lwe.py
  prompts/
    vanilla.py                 # vanilla judge prompt
    lwe_prompts.py             # SSP / LWE prompt templates
  utils/
    dataset.py                 # PairwiseDataset + dataloader
    utils.py
  requirements.txt
```

</details>

## Citation


If you find this work useful, please cite:

```bibtex
@inproceedings{jwa-etal-2026-becoming,
  title = {Becoming Experienced Judges: Selective Test-Time Learning for Evaluators},
  author = {Jwa, Seungyeon and Ahn, Daechul and Kim, Reokyoung and Kang, Dongyeop and Choi, Jonghyun},
  booktitle = {Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)},
  year = {2026},
  address = {Rabat, Morocco},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2026.eacl-short.50/},
  pages = {697--721}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

All datasets and codebases are used strictly for non-commercial academic research
in compliance with their respective licenses. Users of this repository are responsible
for ensuring compliance with the original dataset licenses.

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
