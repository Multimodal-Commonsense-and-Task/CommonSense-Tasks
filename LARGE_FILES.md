# Omitted large files and directories

This repository keeps paper implementations and lightweight configuration files while omitting datasets, generated outputs, checkpoints, and bulky simulation assets that should be obtained from their original sources.

## Existing snapshots

- `Budgeted-CL/budgeted_CL_CIL/collections/`
- `Budgeted-CL/budgeted_CL_LLAVA/collections/`
- `Budgeted-CL/budgeted_CL_LLAVA/$SCRATCH/`
- `FLARE/MMP_results/`
- `ISR-DPO/llava/serve/examples/*.mp4`
- `FALCON/results/`

## Newly added snapshots

- `MoRAS/datasets/` — safety, utility, and calibration datasets (about 637 MB in the upstream tree)
- `OASIS/importance/` — generated importance-analysis results (about 206 MB)
- `GenOL/collections/` — generated dataset collections (about 13 GB)
- `GenOL/backup/`, `GenOL/dataset/`, and `GenOL/dino_resnet50_pretrain.pth`
- `GenOL/nameonly/prompt_generation/{prompts,diversity_prompts,quali_prompts}/` — generated prompt corpora (about 285 MB)
- `DART/third_party/aloha/aloha2/workcell_v2/Aloha workcell v2.stl` — 3D CAD asset (about 93 MB)
- `DART/third_party/libero/libero/libero/assets/` and `DART/third_party/libero-view/libero/libero/assets/` — duplicated LIBERO simulation assets (about 416 MB each)

See the upstream repository READMEs for dataset, checkpoint, and asset download instructions.
