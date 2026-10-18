# Internet Explorer

## Related directories
- `nameonly/internet_explorer/generate_descriptors_gpt_4o.py`
- `nameonly/internet_explorer/download.py`

## Overall Pipeline
1. Generate Descriptors
2. Download Images (web crawler)
3. Subsample, filter, and run experiments similar to the existing method

## Generate Descriptors
- Use `generate_descriptors_gpt_4o.py`
- Change the OpenAI key.
- `count_dict`: Path to the concept count dictionary. (ImageNet, DomainNet, etc.)
    If you want to generate descriptors for a new dataset, you need to create a new count dictionary. (`classes.py`)
- `result_json_path`: Path to save the result of the descriptors JSON.
- `error_txt_path`: Path to save the error log.


## Download Images
- `count_dict`: Name of the concept count dictionary. (ImageNet, DomainNet, etc.)
- `descriptors`: Name of the descriptors JSON file.
- `target_dir`: Name of the target directory to save the downloaded images.
- `increase_ratio`: Ratio of images to download compared to existing dataset (default: 1.15)
- `start_index`, `end_index`: Range of concepts to process

## Run Experiments
- Since IE is a web crawling baseline, the subsequent subsampling, filtering, etc. can be performed similarly to the existing experiments.
