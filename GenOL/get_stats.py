# How to add new dataset config
# 1. get_stats.py -> automatically append mean, std to json file (for augmentation)
# 2. upload dataset to gdrive & modify dataset sh file (PACS_final_grive.sh...)
# 3. Create json (make_collections.py) & move all jsons files to collections/ dir
# 4. modify ex.sh

import os
import cv2
import numpy as np
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG']

transform = transforms.Compose([
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            image = self.transform(image)

        return image

def get_stat(image_root_dir):
    # Calculate mean and std for each channel

    # Get all images recursively
    images = []
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith(tuple(image_exts)):
                images.append(os.path.join(root, file))

    dataset = CustomDataset(images)
    image_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Found {len(dataset)} images")
    # Calculate mean and std
    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    
    # loop through images
    for inputs in tqdm(image_loader):
        inputs = inputs.float() / 255
        psum += inputs.sum(axis=[0, 1, 2])
        psum_sq += (inputs**2).sum(axis=[0, 1, 2])
    
    # pixel count
    count = len(dataset) * 224 * 224

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))

    return {'mean': total_mean.tolist(), 'std': total_std.tolist()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get mean and std for images')
    parser.add_argument('-r', '--image_root_dir', type=str, help='Root directory of images')
    args = parser.parse_args()
    
    last_part = os.path.basename(os.path.normpath(args.image_root_dir))
    
    dataset_list = ["PACS_final", "DomainNet", "cct", "NICO", "cifar10", "ImageNet_200", "ImageNet_400", "ImageNet_full", "CUB_200", "birds31"]
    dataset_name = next((name for name in dataset_list if last_part.startswith(name)), None)
    type_name = last_part[len(dataset_name) + 1:]
    print(f"Detected dataset name: {dataset_name}, type name: {type_name}")

    # Get mean and std
    result = get_stat(args.image_root_dir)
    
    # Write result stats to json file
    json_path = './utils/data_statistics.json'
    with open(json_path, 'r') as f:
        data_statistics = json.load(f)

    mean = (result['mean'][0], result['mean'][1], result['mean'][2])
    std = (result['std'][0], result['std'][1], result['std'][2])
    print(f"Mean: ({result['mean'][0]:.8f}, {result['mean'][1]:.8f}, {result['mean'][2]:.8f})")
    print(f"Std: ({result['std'][0]:.8f}, {result['std'][1]:.8f}, {result['std'][2]:.8f})")
    
    if dataset_name not in data_statistics['mean']:
        data_statistics['mean'][dataset_name] = {}
        data_statistics['std'][dataset_name] = {}
    
    data_statistics['mean'][dataset_name][type_name] = mean
    data_statistics['std'][dataset_name][type_name] = std

    with open(json_path, 'w') as f:
        json.dump(data_statistics, f)