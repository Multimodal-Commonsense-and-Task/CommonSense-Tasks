import os

# Script file name
script_file = "ex.sh"

# Set variables
MODE = "er"
MODEL_NAME = "resnet18"
DATASET = "PACS_final"

TYPES = [
    "CLIP_CurvMatch_10_0_0001",
    "CLIP_CurvMatch_25_0_0001",
    "CLIP_CurvMatch_50_0_0001",
    "CLIP_Glister_10_0_0001",
    "CLIP_Glister_25_0_0001",
    "CLIP_Glister_50_0_0001",
    "CLIP_GradMatch_10_0_0001",
    "CLIP_GradMatch_25_0_0001",
    "CLIP_GradMatch_50_0_0001",
    "CLIP_Uncertainty_10_0_0001",
    "CLIP_Uncertainty_25_0_0001",
    "CLIP_Uncertainty_50_0_0001",
]

# Generate sbatch commands and submit jobs
for type_name in TYPES:
    for seed in ['1', '2', '3', '4', '5']:
        sbatch_command = f"sbatch {script_file} {MODE} {MODEL_NAME} {DATASET} {type_name} {seed}"
        print(f"Submitting job with command: {sbatch_command}")
        os.system(sbatch_command)
