## 1. Installation

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
cd ../
pip install -r requirements_sdxl.txt
pip install peft==0.15.0
pip install datasets
```

## 2. Prepare the Dataset

Subsample the manually annotated (MA) dataset to create a new folder under `db/`, using **k representative images per class**. The structure should look like this:

```
db/
└── PACS_final_train_ma_3/
    ├── bird/
    │   ├── *.jpg
    │   ├── *.jpg
    │   └── *.jpg
    ├── dog/
    │   ├── *.jpg
    │   ├── *.jpg
    │   └── *.jpg
    └── ...
```

Each subfolder should correspond to a class label (e.g., `bird`, `dog`) and contain k images.

## 3. Generate Metadata

Run the following script to generate the necessary metadata for training:

```bash
python generate_metadata.py --image_dir ./PACS_final_train_ma_3
```

## 4. Training

- Edit `train.sh`:
  - Set `IMAGE_DIR` to the dataset folder name (e.g., `PACS_final_train_ma_3`)
  - Set `OUTPUT_DIR` to the desired output directory for saving the checkpoint

Then run:

```bash
bash train.sh
```

## 5. Image Generation

- Open `nameonly/generate_twostage/run.sh`
- Set the `LORA_PATH` to the checkpoint file from training:

```bash
LORA_PATH="[checkpoint_dir]/pytorch_lora_weights.safetensors"
```

Then run the script to generate images.