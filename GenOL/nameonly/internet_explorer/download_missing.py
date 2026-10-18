import os
import sys
import json
import math
import shutil
from tqdm import tqdm
from better_bing_image_downloader import downloader
from unittest.mock import patch
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(target_dir)
from classes import get_count_dict, get_discription_dict

dataset = "ImageNet"
count_dict = get_count_dict(dataset)
descriptors = f"{dataset}_descriptors.json"
target_dir = f"{dataset}_internet_explorer_missing"
target_index = [201, 9, 319, 379] 
# [105, 12, 127, 153, 169, 2, 201, 249, 258, 269, 276, 277, 288, 299, 318, 319, 329, 330, 349, 354, 364, 366, 367, 368, 369, 375, 379, 439, 642, 655, 699, 850, 9]
increase_ratio = 1.15
concepts = list(count_dict.keys())
concepts = np.array(concepts)[target_index]
expand_ratio = 6
description_dict = get_discription_dict(dataset) if dataset=="ImageNet" else None


image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG']

with open(descriptors, 'r') as f:
    descriptors_dict = json.load(f)
    

for concept in tqdm(concepts):
    # concept = concept.replace('', '')
    download_path = os.path.join(target_dir, concept)
    current_count = 0
    min_images = int(count_dict[concept] * increase_ratio * expand_ratio)
    descriptors_list = descriptors_dict[concept]
    download_per_descriptor = int(math.ceil(min_images / len(descriptors_list))) * expand_ratio
    if os.path.exists(download_path) and len(os.listdir(download_path)) >= min_images:
        print(f"Skipping {concept} as it already has {min_images} images")
        continue
    
    print(f"Downloading images for {concept} with a minimum of {min_images} images")
    print(f"Descriptor len {len(descriptors_list)} download_per_descriptor {download_per_descriptor}")
    
    for i, descriptor in enumerate(descriptors_list):
        # if current_count >= min_images:
        #     break

        if dataset=='ImageNet':
            query_string = f"{descriptor} {description_dict[concept]}"
            print(f"downloading {description_dict[concept]}")
        else:
            query_string = f"{descriptor} {concept}"
        with patch('builtins.input', return_value='N'):
            downloader(query_string, limit=download_per_descriptor, output_dir=download_path, adult_filter_off=True, 
                    timeout=60, filter="", verbose=False, badsites= [], name=f"Image_{i}")
            
        # Count the number of images recursively
        current_count = sum([len(files) for root, dirs, files in os.walk(download_path)])
        print(f"Current count for {concept}: {current_count} / {min_images}")

    print(f"Finished downloading images for {concept}")
    
    # Reorganize the image directory
    image_num = 0
    dirs = [d for d in os.listdir(download_path) if os.path.isdir(os.path.join(download_path, d))]
    # Move all images to the root directory
    for d in dirs:
        images = [f for f in os.listdir(os.path.join(download_path, d)) if os.path.isfile(os.path.join(download_path, d, f))]
        images = [f for f in images if os.path.splitext(f)[1] in image_exts]
        for image in images:
            shutil.copy(os.path.join(download_path, d, image), os.path.join(download_path, f"{str(image_num).zfill(5)}{os.path.splitext(image)[1]}"))
            image_num += 1
    
    # Remove all subdirectories
    for d in dirs:
        shutil.rmtree(os.path.join(download_path, d))

