# Subsampling, Filtering, RMD-Ensemble Process

The overall pipeline follows these steps:
1. Subsample the generated images by 1.1x and save them to the `<IMAGE_DIR>_subsampled` folder.
2. Filter the subsampled images and save them to the `<IMAGE_DIR>_subsampled_filtered` folder.
3. Calculate the RMD scores and save them as a JSON file.
4. Perform ensemble based on the RMD scores.

---

## `classes.py`
- The dataset names used in subsampling, filtering, and RMD-ensemble must be defined in `classes.py`.
- This file stores class names and the number of images per class for each dataset.
- Currently, due to module import issues, this file is copied across multiple folders, but it needs to be resolved:
    - `./nameonly/classes.py`
    - `./nameonly/crawling/classes.py`
    - `./nameonly/prompt_generation/classes.py`
    - `./nameonly/generate_twostage/classes.py`

---

## Subsampling
- Use `./nameonly/subsample.py`.
- Example usage:
    ```bash
    python subsample.py -d ImageNet_full -s ./IMAGENET_DIR/ImageNet_LE_sdxl -r 1.1
    ```

---

## Filtering
- Use `./nameonly/filter_images.py`.
- Images are automatically filtered to match the MA size.
- Anaconda environment name: `generate`.
- Example usage:
    ```bash
    python filter_images.py -d ImageNet_full -s ./IMAGENET_DIR/ImageNet_LE_sdxl_subsampled
    ```

By repeating this process for each model, the following folder structure will be created. Each directory contains filtered images:
- `./IMAGENET_DIR/ImageNet_LE_sdxl_subsampled_filtered`
- `./IMAGENET_DIR/ImageNet_LE_floyd_subsampled_filtered`
- `./IMAGENET_DIR/ImageNet_LE_cogview2_subsampled_filtered`
- `./IMAGENET_DIR/ImageNet_LE_sd3_subsampled_filtered`
- `./IMAGENET_DIR/ImageNet_LE_auraflow_subsampled_filtered`

---

## RMD-Ensemble (RMD Score Calculation)
- Use `./nameonly/calculate_rmd_scores.py`.
- Modify the YAML file in the `./nameonly/configs` folder:
    - Update the dataset name (it must be defined in `classes.py`).
    - Set `use_ma_size` to `False` (this option can be ignored for now).
    - Specify the paths to the directories containing the filtered images.
    - Set the `json_save_path` to the desired output path for the JSON file containing RMD scores.
- Example usage:
    ```bash
    python calculate_rmd_scores.py --config_path ./configs/ImageNet.yaml
    ```

---

## RMD Ensemble
- Use `./nameonly/sample_with_rmd_scores_samplewise.py`.
- Specify the following options:
  - `normalize`, `clip`, `lower_percentile`, `upper_percentile`.
  - Specify ensemble methods: `equalweight`, `TopK`, `BottomK`, `INVERSE`, `TEMPERATURE`.
- Key parameters:
  - `base_path`: Path to the directory containing filtered images (parent directory of model-specific folders).
  - `json_path`: Path to the JSON file containing RMD scores.
  - `target_path`: Path to save the ensemble results.
  - `count_dict`: A dictionary containing the number of images per class, as defined in `classes.py`.
    - Use the `get_count_value_from_string` function to automatically fetch the count value based on the `base_path` name.
    - For now, since the ImageNet size is being varied, the count must be specified manually, e.g., `count_dict = ImageNet_full`.

- Example usage:
    ```bash
    python sample_with_rmd_scores_samplewise.py
    ```

- The results will be saved to the `target_path` folder, with the number of images matching the MA size as defined in `classes.py`.
