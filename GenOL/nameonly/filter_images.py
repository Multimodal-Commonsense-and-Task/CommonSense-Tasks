import os
import argparse
from utils import get_text_image_similarity
from transformers import CLIPProcessor, CLIPModel
import shutil
import torch
from tqdm import tqdm
from classes import *
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default=None)
parser.add_argument('-s', '--image_root_dir', type=str)
parser.add_argument('-t', '--output_prefix', type=str, default='_filtered')
parser.add_argument('-p', '--image_prefix', type=str, default='')
parser.add_argument('--start_class', type=int, default=0)
parser.add_argument('--end_class', type=int)
parser.add_argument('--ratio', type=float, default=1)
parser.add_argument('--remove_ratio', type=float, default=0)
parser.add_argument('--lowest_similarity_path', type=str, default='None')

args = parser.parse_args()

# Define default configurations for datasets
replacements = {
    "PACS": "PACS",
    "cct": "cct",
    "DomainNet": "DomainNet",
    "NICO": "NICO",
    "cifar10": "cifar10",
    "ImageNet": "ImageNet",
}

dataset_mapping = {'DomainNet': DomainNet_count, 'officehome': officehome_count, 'PACS': PACS_count, 
                   'birdsnap': birdsnap_count, 'cifar10': cifar10_count, 'aircraft': aircraft_count,
                   'food101': food101_count, 'cct': cct_count, 'pacs_sdxl': pacs_sdxl_count, 
                   'pacs_dalle2': pacs_dalle2_count, 'pacs_deepfloyd': pacs_deepfloyd_count,
                   'pacs_cogview2': pacs_cogview2_count, 'pacs_sdxl_new': pacs_sdxl_new_count,
                   'pacs_dalle2_new': pacs_dalle2_new_count, 'NICO': NICO_count,
                   'ImageNet': ImageNet_count, 'CUB_200': CUB_200_count}

# Find dataset name
if args.dataset is None:
    for pattern, replacement in replacements.items():
        if pattern.lower() in args.image_root_dir.lower():
            print(f"Dataset not specified. Detected dataset: {replacement}")
            args.dataset = replacement
            break

sample_num_dict = get_count_dict(args.dataset)
# sample_num_dict = dataset_mapping[args.dataset]
print(f"Sample num dict: {sample_num_dict}")
if "ImageNet" in args.dataset:
    print("Detected dataset: ImageNet")
    imagenet = True
else:
    imagenet = False

device = "cuda" if torch.cuda.is_available() else "cpu"
# Check args.end_class
if args.end_class is None:
    args.end_class = len(os.listdir(args.image_root_dir)) - 1

source_path = Path(args.image_root_dir)
output_dir = os.path.join(source_path.parent, f"{source_path.name}{args.output_prefix}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load the images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPEG']
class_to_paths_dict = {}

classes = sample_num_dict.keys()

for class_name in classes:
    class_to_paths_dict[class_name] = []
    class_dir = os.path.join(args.image_root_dir, class_name)
    for root, dirs, files in os.walk(class_dir):
        for file in files:
            if file.endswith(tuple(image_extensions)):
                class_to_paths_dict[class_name].append(os.path.join(root, file))
    
if args.end_class is None:
    args.end_class = len(classes) - 1

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Preprocess the images
for class_name, image_paths in tqdm(class_to_paths_dict.items()):
    # if os.path.exists(os.path.join(args.output_dir, class_name)):
    #     print(f"Skipping {class_name} as it already exists")
    #     continue
    print('Processing class:', class_name)
    if args.remove_ratio == 0:
        sample_num = int(sample_num_dict[class_name] * args.ratio)
    else:
        # Just remove a certain ratio of images
        sample_num = len(image_paths) - int(len(image_paths) * args.remove_ratio)
    
    if len(image_paths) > sample_num:
        prompt = f"A photo of {class_name}"
        if not imagenet:
            formatted_class_name = class_name.replace('_', ' ')
        else:
            formatted_class_name = ImageNet_description[class_name].replace('_', ' ')
        
        # Add article to prompt
        if formatted_class_name[0] in ['a', 'e', 'i', 'o', 'u']:
            prompt = f"A photo of an {formatted_class_name}"
        else:
            prompt = f"A photo of a {formatted_class_name}"

        print(f"Filtering prompt: {prompt}")
        if image_paths == []:
            raise ValueError(f"No images found for class {class_name}")
        
        image_text_similarity = get_text_image_similarity(prompt, image_paths, processor, model)
        
        # Get top-k imag_paths
        top_k_indices = image_text_similarity.argsort(descending=True)[:sample_num]
        top_k_image_paths = [image_paths[i] for i in top_k_indices]

        # Print the image path of the lowest similarity
        lowest_samples = image_text_similarity.argsort()[:10]
        if args.lowest_similarity_path != 'None':
            print(f"Saving lowest similarity images to {args.lowest_similarity_path}")
            if not os.path.exists(args.lowest_similarity_path):
                os.makedirs(args.lowest_similarity_path)
            for i, index in enumerate(lowest_samples):
                image_path = image_paths[index]
                image_name = os.path.basename(image_path)
                output_image_path = os.path.join(args.lowest_similarity_path, f"{class_name}_{image_name}")
                shutil.copy(image_path, output_image_path)
    else:
        # No filtering (only cp)
        print(f"[WARNING] - sample num from dict: {sample_num}, from dir: {len(image_paths)}")
        top_k_image_paths = image_paths
    
    # Copy the top-k images to the output directory
    output_class_dir = os.path.join(output_dir, class_name)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    image_id = 0
    for image_path in tqdm(top_k_image_paths, desc='Copying images'):
        image_name = str(image_id).zfill(6) + '.png'
        # Add prefix to image name
        if args.image_prefix != '':
            image_name = args.image_prefix + image_name
        output_image_path = os.path.join(output_class_dir, image_name)
        shutil.copy(image_path, output_image_path)
        image_id += 1

