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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--image_root_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--pickle_path', type=str)
parser.add_argument('--image_prefix', type=str, default='')
parser.add_argument('--ratio', type=float, default=0.1)
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

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Load pickle file
with open(args.pickle_path, 'rb') as f:
    rmd_scores = pickle.load(f)

# Convert the RMD score mapping ((model, cls):(path, score) -> (path:score))
# 필요한 것은 path (./datasets/PACS/final/PACS_final_web_RMD_temp_0_5/house/flickr_000144.png) 이렇게 주어졌을 때
# RMD score mapping에서 이 sample의 RMD score를 찾는 것이다. 이를 위해 일단 RMD mapping을 위 path 형태로 바꿔주면 된다.
path_score_mapping = {}
for k, v in rmd_scores.items():
    model, class_name = k
    for path, score in v:
        # final dataset (실험 돌린 것) 형태에 맞게 path를 바꿔줘야 함
        final_dataset_path = f"./datasets/PACS/final/PACS_final_web_RMD_temp_0_5/{class_name}/{os.path.basename(path)}"
        path_score_mapping[final_dataset_path] = score

# Load the images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPEG']
class_to_paths_dict = {}

classes = sample_num_dict.keys()

for class_name in classes:
    class_to_paths_dict[class_name] = []
    class_dir = os.path.join(args.image_root_dir, class_name)
    for root, dirs, files in os.walk(class_dir):
        for file in files:
            if file.endswith(tuple(image_extensions)) and '._' not in file:
                class_to_paths_dict[class_name].append(os.path.join(root, file))

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Preprocess the images
for class_name, image_paths in tqdm(class_to_paths_dict.items()):
    # if os.path.exists(os.path.join(args.output_dir, class_name)):
    #     print(f"Skipping {class_name} as it already exists")
    #     continue
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

    # Get RMD score of normal / ood samples
    RMD_normal = [path_score_mapping[path] for path in normal_samples]
    RMD_ood = [path_score_mapping[path] for path in ood_samples]


    plt.figure()
    # sns.kdeplot(RMD_normal, shade=True, color='green', label='Normal samples')
    # sns.kdeplot(RMD_ood, shade=True, color='red', label='OOD samples')
    sns.histplot(RMD_normal, color='green', kde=True, alpha=0.5, label='Normal samples')
    sns.histplot(RMD_ood, color='red', kde=True, alpha=0.5, label='OOD samples')
    plt.legend()
    figure_name = os.path.join(args.visualize_path, f"{class_name}_{str(args.ratio).replace('.', '_')}.png")
    plt.savefig(figure_name)
    
    # # Print the image path of the lowest similarity
    # lowest_samples = image_text_similarity.argsort()[:10]
    # if args.lowest_similarity_path != 'None':
    #     print(f"Saving lowest similarity images to {args.lowest_similarity_path}")
    #     if not os.path.exists(args.lowest_similarity_path):
    #         os.makedirs(args.lowest_similarity_path)
    #     for i, index in enumerate(lowest_samples):
    #         image_path = image_paths[index]
    #         image_name = os.path.basename(image_path)
    #         output_image_path = os.path.join(args.lowest_similarity_path, f"{class_name}_{image_name}")
    #         shutil.copy(image_path, output_image_path)

    # # Copy the top-k images to the output directory
    # output_class_dir = os.path.join(args.output_dir, class_name)
    # if not os.path.exists(output_class_dir):
    #     os.makedirs(output_class_dir)

    # image_id = 0
    # for image_path in tqdm(top_k_image_paths, desc='Copying images'):
    #     image_name = str(image_id).zfill(6) + '.png'
    #     # Add prefix to image name
    #     if args.image_prefix != '':
    #         image_name = args.image_prefix + image_name
    #     output_image_path = os.path.join(output_class_dir, image_name)
    #     shutil.copy(image_path, output_image_path)
    #     image_id += 1

