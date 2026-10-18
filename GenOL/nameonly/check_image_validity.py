#!/usr/bin/env python3

import cv2
import os
from classes import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default=None)
parser.add_argument('-s', '--source_path', type=str, required=True)
parser.add_argument('-t', '--threshold_ratio', type=float, default=1.0)
args = parser.parse_args()

RED = "\033[31m"
RESET = "\033[0m"

replacements = {
    "PACS": "PACS",
    "cct": "cct",
    "DomainNet": "DomainNet",
    "NICO": "NICO",
    "cifar10": "cifar10",
    "ImageNet": "ImageNet",
    "CUB_200": "CUB_200",
    "birds31": "birds31"
}

# Find dataset name
if args.dataset is None:
    for pattern, replacement in replacements.items():
        if pattern.lower() in args.source_path.lower():
            print(f"Dataset not specified. Detected dataset: {replacement}")
            args.dataset = replacement
            break

def check_class_names(directory_path, class_names_dict):
    class_names = os.listdir(directory_path)
    class_names = [cls for cls in class_names if not cls.endswith('.json')]
    for cls in class_names:
        if cls not in class_names_dict.keys():
            print(f"Class {cls} not found in class names dictionary.")
            raise ValueError(f"Class {cls} not found in class names dictionary.")
    # Find classes in directory that are not in class names dictionary
    not_exist = list(set(class_names_dict.keys()) - set(class_names))
    
    # Find indices of classes in directory that are not in class names dictionary
    class_names_ordered = list(class_names_dict.keys())
    not_exist_indices = sorted([class_names_ordered.index(cls) for cls in not_exist])
    print(f"Not found classes: {not_exist}, corresponding indices: {not_exist_indices}")
    print(f"Number of not found classes: {len(not_exist)}")
    
    assert len(class_names) == len(class_names_dict.keys()), "Number of classes in directory does not match number of classes in class names dictionary."
    
def check_image_size(image_path, size):
    image = cv2.imread(image_path)
    if image.shape[:2] != size:
        print(f"Image {image_path} has shape {image.shape} which does not match the expected shape {size}")
        return False
    return True

def convert_images_in_directory(directory_path):
    print(f"Checking corrupted images in {directory_path}")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPEG']
    corrupted_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                # print(f"Processing {file_path}")
                try:
                    # print(f"Processing {file_path}")
                    image = cv2.imread(file_path)
                    if image is None:
                        raise ValueError("Image could not be read, possibly due to a format issue or file corruption.")
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    corrupted_files.append(file_path)

    print(f"Corrupted files: {corrupted_files}")

    if len(corrupted_files) == 0:
        return True
    else:
        return False

# CHANGE THIS!!!
directory_path = args.source_path


dataset_mapping = {'DomainNet': DomainNet_count, 'officehome': officehome_count, 'PACS': PACS_count, 
                   'birdsnap': birdsnap_count, 'cifar10': cifar10_count, 'aircraft': aircraft_count,
                   'food101': food101_count, 'cct': cct_count, 'pacs_sdxl': pacs_sdxl_count, 
                   'pacs_dalle2': pacs_dalle2_count, 'pacs_deepfloyd': pacs_deepfloyd_count,
                   'pacs_cogview2': pacs_cogview2_count, 'pacs_sdxl_new': pacs_sdxl_new_count,
                   'pacs_dalle2_new': pacs_dalle2_new_count, 'NICO': NICO_count,
                   'ImageNet': ImageNet_count, 'CUB_200': CUB_200_count}

class_names_dict = get_count_dict(args.dataset)
# class_names_dict = dataset_mapping[args.dataset]
dir_cls_count_dict = {cls: len(os.listdir(os.path.join(directory_path, cls))) for cls in os.listdir(directory_path) if not cls.endswith('.json') and os.path.isdir(os.path.join(directory_path, cls))}

print(f"Current threshold ratio: {args.threshold_ratio}")
# Step 1-1: Check class names
for dir_cls in dir_cls_count_dict.keys():
    if dir_cls not in class_names_dict.keys():
        print(f"Class {dir_cls} not found in class names dictionary.")
        raise ValueError(f"Class {dir_cls} not found in class names dictionary.")

# Step 1-2: Check class count
less_than_threshold = []
for cls, count in class_names_dict.items():
    if cls not in dir_cls_count_dict.keys():
        less_than_threshold.append(cls)
    elif dir_cls_count_dict[cls] < count * args.threshold_ratio:
        less_than_threshold.append(cls)

enough = len(less_than_threshold) == 0
exact_match = dir_cls_count_dict == class_names_dict

# Step 2: Check image size
# Choose all images in the first class
class_names = [cls for cls in os.listdir(directory_path) if not cls.endswith('.json')]
class_name = class_names[0]

for image in os.listdir(os.path.join(directory_path, class_name)):
    if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png') or image.endswith("JPEG"):
        # Check
        image_path = os.path.join(directory_path, class_name, image)
        # print(f"Checking image size...")
        image_size_correct = check_image_size(image_path, (224, 224))
        if not image_size_correct:
            print(f"{RED}WARNING: Image {image_path} does not have the correct size.{RESET}")
            break

print(f"Count enough (over threshold): {enough}")
print(f"Count exact match: {exact_match}")
# print(f"Image size correct: {image_size_correct}")

if not enough:
    print(f"{RED}Classes with less than threshold count: {less_than_threshold}{RESET}")
    raise ValueError(f"Classes with less than threshold count: {less_than_threshold}")

# Step 3: Check image validity
convert_correct = convert_images_in_directory(directory_path)
