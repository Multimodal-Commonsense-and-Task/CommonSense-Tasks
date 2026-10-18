# Attention Misses Visual Risk: Risk-Adaptive Steering for Multimodal Safety Alignment (ECCV 2026)

Official implementation of **"Attention Misses Visual Risk: Risk-Adaptive Steering for Multimodal Safety Alignment"** (ECCV 2026).
[[Paper]](https://arxiv.org/abs/2510.13698)

[`Jonghyun Park`](https://jonghyun813.github.io/profile/) | [`Minhyuk Seo`](https://dbd05088.github.io/) | [`Chaewon Yeo`](https://github.com/Chaewon-Yeo) | [`Jonghyun Choi`](https://ppolon.github.io/)†

<sub><span style="color:gray">† Corresponding author</span></sub>

## Installation

```bash
conda create -n moras_llava python=3.10 -y
conda activate moras_llava
python -m pip install -r requirements.txt
```

## Dataset Preparation

Place all datasets under the repository-level `datasets/` directory. The paths below are relative to the repository root.

| Benchmark | Required path |
| --- | --- |
| SPA-VL | Downloaded automatically from Hugging Face (`sqrti/SPA-VL`) |
| FigStep | `datasets/safety/figstep/` (place the benchmark images directly in this directory) |
| MM-SafetyBench | `datasets/safety/mmsafety/processed_questions/<scenario>.json` and `datasets/safety/mmsafety/images/<scenario>/SD_TYPO/<id>.jpg` |
| JOOD | `datasets/safety/AdvBenchM/jood_dataset.json`; image paths in the JSON must resolve relative to the repository root |
| MM-Vet | `datasets/utility/mm-vet/mm-vet.json` and `datasets/utility/mm-vet/images/` |
| GQA | `datasets/utility/gqa/llava_gqa_testdev_balanced.jsonl` and `datasets/utility/gqa/images/` |
| MME | `datasets/utility/MME/llava_mme.jsonl` and `datasets/utility/MME/MME_Benchmark_release_version/` |
| ScienceQA | `datasets/utility/scienceqa/llava_test_CQM-A.json` and `datasets/utility/scienceqa/images/test/` |

For VLSafe-based calibration with `parameters.get_baseline_cossim`, use:

```text
datasets/calibration/VLSafe/
├── images/
└── train/
    ├── VLSafe_harmlessnss_alignment.jsonl
    └── vlsafe_convert_safe.jsonl
```

## Parameter Calibration

Run the following command from the `LLaVA` directory:

```bash
cd LLaVA
```

Generate the unsafe-state prototypes:

```bash
python -m parameters.get_unsafe_states --eval_model=llava_v1_5_7b
```

Compute the baseline cosine similarity and sigmoid parameters:

```bash
python -m parameters.get_baseline_cossim --eval_model=llava_v1_5_7b
```

## Inference

After preparing the benchmark datasets, run inference from the `LLaVA` directory. For example, run MoRAS on SPA-VL with:

```bash
python -m eval.spavl --eval_model=llava_v1_5_7b
```

Other supported safety benchmarks can be run in the same way:

```bash
python -m eval.figstep --eval_model=llava_v1_5_7b
python -m eval.mmsafety --eval_model=llava_v1_5_7b
python -m eval.jood --eval_model=llava_v1_5_7b
```

For utility evaluation:

```bash
python -m eval.mmvet --eval_model=llava_v1_5_7b
python -m eval.gqa --eval_model=llava_v1_5_7b
python -m eval.mme --eval_model=llava_v1_5_7b
python -m eval.sciqa --eval_model=llava_v1_5_7b
```

Outputs are saved under `LLaVA/results/<eval_model>/moras/`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{park2026attention,
  title={Attention Misses Visual Risk: Risk-Adaptive Steering for Multimodal Safety Alignment},
  author={Park, Jonghyun and Seo, Minhyuk and Yeo, Chaewon and Choi, Jonghyun},
  booktitle={European Conference on Computer Vision},
  pages={38--56},
  year={2026},
  organization={Springer}
}
```

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
