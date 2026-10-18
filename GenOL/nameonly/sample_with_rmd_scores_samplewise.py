import os, random, shutil
import json
import pickle
import numpy as np
from classes import *
from utils import softmax_with_temperature
from tqdm import tqdm
from pathlib import Path

# NORMALIZATION, CLIPPING
normalize = True # Fix
clip = True # Fix
lower_percentile = 5.0 # 5.0
# lower_percentile = 2.5 # 5.0
upper_percentile = 95.0 # 95.0
# upper_percentile = 97.5 # 95.0

equalweight = False
TopK = False
BottomK = False
INVERSE = False
TEMPERATURE = 0.5

# IMPORTANT
base_path = '/home/user/smh/New_Nameonly_CL/nameonly/generate_twostage/generated_datasets/ImageNet_final'
json_path = './RMD_scores/ImageNet_ours_full_sdxl_floyd_cogview2_sd3_auraflow.json'
target_path = './raw_datasets/iclr_rebuttal/ImageNet/ImageNet_ours_full_sdxl_floyd_cogview2_sd3_auraflow'
# IMPORTANT

count_dict = get_count_value_from_string(base_path)
count_dict = ImageNet_full_count

with open(json_path, 'r') as f:
    RMD_scores = json.load(f)

# Parse RMD pickle file to get PATH dict
models = list(RMD_scores.keys())
PATH_dict = {}
for model in models:
    first_class = next(iter(RMD_scores[model]))
    relative_path = RMD_scores[model][first_class][0]['image_path'] # Get the first item
    PATH_dict[model] = os.path.join(base_path, str(Path(relative_path).parents[1]))

# Shuffle the images
for model in PATH_dict.keys():
    for cls in count_dict.keys():
        random.shuffle(RMD_scores[model][cls])

ensembled_images = {cls: [] for cls in count_dict.keys()}
model_class_selected_counter = {model: {cls: 0 for cls in count_dict.keys()} for model in PATH_dict.keys()}

for cls in tqdm(list(count_dict.keys())):
    # For top-k, get list of [(model, image_path, score), ...]
    image_rmd_scores = []
    for model, cls_images_dict in RMD_scores.items():
        cls_images = cls_images_dict[cls]
        for image_dict in cls_images:
            image_path = os.path.join(base_path, image_dict['image_path'])
            image_rmd_scores.append((model, image_path, image_dict['score']))

    # Get sample_path -> (model, score) mapping
    sample_model_RMD_mapping = {}
    for sample in image_rmd_scores:
        sample_model_RMD_mapping[sample[1]] = sample[0], sample[2] # model, score

    if TopK:
        sorted_data = sorted(sample_model_RMD_mapping.items(), key=lambda item: item[1][1], reverse=True)
        chosen_samples = [sample[0] for sample in sorted_data[:count_dict[cls]]]
    elif BottomK:
        sorted_data = sorted(sample_model_RMD_mapping.items(), key=lambda item: item[1][1], reverse=False)
        chosen_samples = [sample[0] for sample in sorted_data[:count_dict[cls]]]
    elif equalweight:
        probabilities = [1 / len(PATH_dict)] * len(PATH_dict)
        chosen_samples = []
        while True:
            chosen_model = random.choices(models, weights=probabilities, k=1)[0]
            if len(RMD_scores[chosen_model][cls]) > 0:
                chosen_image = RMD_scores[chosen_model][cls].pop()
                chosen_image_path = os.path.join(base_path, chosen_image['image_path'])
                chosen_samples.append(chosen_image_path)
            if len(chosen_samples) == count_dict[cls]:
                print(f"Break for class {cls} with {len(chosen_samples)} images")
                break
    else:
        # Normalize and clip RMD scores
        scores = np.array([score[1] for score in sample_model_RMD_mapping.values()])
        # mean = np.mean(scores); std = np.std(scores)
        
        if clip:
            lower_clip = np.percentile(scores, lower_percentile)
            upper_clip = np.percentile(scores, upper_percentile)
            # print(f"Lower clip: {lower_clip}, Upper clip: {upper_clip}")
            clipped_scores = np.clip(scores, lower_clip, upper_clip)
            if normalize:
                mean = np.mean(clipped_scores); std = np.std(clipped_scores)
                result_scores = (clipped_scores - mean) / std
            else:
                result_scores = clipped_scores
        else:
            result_scores = scores
        
        probabilities = softmax_with_temperature(result_scores, TEMPERATURE)
        if INVERSE:
            # To get the inverse probabilities, first handle the numerical instability
            if np.min(probabilities) < 0:
                probabilities -= np.min(probabilities)
            # Handle devision by zero
            if np.sum(probabilities) == 0:
                raise ValueError("All probabilities are zero")
            probabilities = 1 / probabilities
            probabilities /= np.sum(probabilities) # Normalize the probabilities

        chosen_samples = np.random.choice(list(sample_model_RMD_mapping.keys()), size=count_dict[cls], replace=False, p=probabilities)
    
    # Update the result dictionary and counter
    for sample_path in chosen_samples:
        ensembled_images[cls].append({'model': sample_model_RMD_mapping[sample_path][0], 'image': sample_path})
        model_class_selected_counter[sample_model_RMD_mapping[sample_path][0]][cls] += 1


# # Check the number of images selected for each model, for each class
# for model, class_counter in model_class_selected_counter.items():
#     print(f"Model {model} selected for each class:")
#     print(class_counter)

# Sanity check the number of images for each class
for cls, images in ensembled_images.items():
    if len(images) != count_dict[cls]:
        raise ValueError(f"Class {cls} has {count_dict[cls]} images but {len(images)} images are selected")

# Copy all the images to the target path
# Remove target path if already exists
if os.path.exists(target_path):
    raise OSError(f"Target path already exists! - {target_path}")

for cls, images in ensembled_images.items():
    image_counter = 0
    target_cls_path = os.path.join(target_path, cls)
    os.makedirs(target_cls_path, exist_ok=True)
    for img in images:
        model = img['model']; image_name = img['image']
        new_image_name = f"{model}_{str(image_counter).zfill(6)}.png"
        image_counter += 1
        shutil.copy(image_name, os.path.join(target_cls_path, new_image_name))

# Print the number of images selected for each model
for model, class_counter in model_class_selected_counter.items():
    # Sum the number of images selected for each class
    total_images = sum(class_counter.values())
    print(f"Model {model} selected for each class: {total_images} images")