# CLIP score로 10%로 outlier를 찾고, 이를 이용해서 RMD를 outlier vs normal plot하기 위한 코드
# 0417에 짠 코드

import os
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_text_image_similarity
from transformers import CLIPProcessor, CLIPModel
import shutil
import torch
from tqdm import tqdm
from classes import *

# Strong filtered
flickr_path = './datasets/PACS/PACS_flickr_filtered'
google_path = './datasets/PACS/PACS_google_filtered'
bing_path = './datasets/PACS/PACS_bing_filtered'
source_paths = [flickr_path, google_path, bing_path]

image_root_dir_rmd = './datasets/PACS/final/PACS_final_web_RMD_temp_0_5'
image_root_dir_topk = './datasets/PACS/final/PACS_final_web_topk'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PACS')
# parser.add_argument('--pickle_path', type=str)
parser.add_argument('--image_prefix', type=str, default='')
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--visualize_path', type=str, default='./vis')

args = parser.parse_args()

dataset_mapping = {'DomainNet': DomainNet_count, 'officehome': officehome_count, 'PACS': PACS_count, 
                   'birdsnap': birdsnap_count, 'cifar10': cifar10_count, 'aircraft': aircraft_count,
                   'food101': food101_count, 'cct': cct_count, 'pacs_sdxl': pacs_sdxl_count, 
                   'pacs_dalle2': pacs_dalle2_count, 'pacs_deepfloyd': pacs_deepfloyd_count,
                   'pacs_cogview2': pacs_cogview2_count, 'pacs_sdxl_new': pacs_sdxl_new_count,
                   'pacs_dalle2_new': pacs_dalle2_new_count, 'ImageNet': ImageNet_count,
                   'CUB_200': CUB_200_count}

sample_num_dict = dataset_mapping[args.dataset]
print(f"Sample num dict: {sample_num_dict}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPEG']
class_to_paths_dict = {}

classes = sample_num_dict.keys()

for class_name in classes:
    class_to_paths_dict[class_name] = []
    for path in source_paths:
        class_dir = os.path.join(path, class_name)
        for root, dirs, files in os.walk(class_dir):
            for file in files:
                if file.endswith(tuple(image_extensions)) and '._' not in file:
                    class_to_paths_dict[class_name].append(os.path.join(root, file))

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Preprocess the images
for class_name, image_paths in tqdm(class_to_paths_dict.items()):
    print('Processing class:', class_name)
    prompt = f"A photo of {class_name}"
    formatted_class_name = class_name.replace('_', ' ')
    # Add article to prompt
    if formatted_class_name[0] in ['a', 'e', 'i', 'o', 'u']:
        prompt = f"A photo of an {formatted_class_name}"
    else:
        prompt = f"A photo of a {formatted_class_name}"

    print(f"Filtering prompt: {prompt}")
    if image_paths == []:
        raise ValueError(f"No images found for class {class_name}")
    image_text_similarity = get_text_image_similarity(prompt, image_paths, processor, model)
    
    # Get normal / ood samples
    sorted_indices = image_text_similarity.argsort(descending=True)
    sorted_image_paths = [image_paths[i] for i in sorted_indices]

    num_normal = int((1 - args.ratio) * len(sorted_indices))
    num_ood = len(sorted_indices) - num_normal

    normal_samples = sorted_image_paths[:num_normal]
    ood_samples = sorted_image_paths[num_normal:]

    # Count the number of samples included in OOD_samples
    ood_samples_basename = [os.path.basename(path) for path in ood_samples]
    rmd_samples = os.listdir(os.path.join(image_root_dir_rmd, class_name))
    topk_samples = os.listdir(os.path.join(image_root_dir_topk, class_name))

    ood_samples_in_rmd_ensemble = set(ood_samples_basename) & set(rmd_samples)
    ood_samples_in_topk_ensemble = set(ood_samples_basename) & set(topk_samples)
    ood_samples_intersect = ood_samples_in_rmd_ensemble & ood_samples_in_topk_ensemble
    print(f"# OOD samples in RMD: {len(ood_samples_in_rmd_ensemble)}")
    print(f"# OOD samples in TopK: {len(ood_samples_in_topk_ensemble)}")
    print(f"# OOD samples intersect: {len(ood_samples_intersect)}")
    print("#"*30)
