import os
from classes import *

target_class = food101_count
target_path = f"./datasets/food101_flickr"
download_threshold = 480

# Read all txt files and check the number of URLs
not_existing_class = []
not_enough_classes = {}
for cls in target_class:
    cls_path = os.path.join(target_path, cls, '00000')
    if not os.path.exists(cls_path):
        not_existing_class.append(cls)
        continue
    else:
        num_images = len(os.listdir(cls_path))
        if num_images < download_threshold:
            print(f"WARNING: Class {cls} has {num_images} images")
            not_enough_classes[cls] = num_images

print(f"Classes that do not have any dir: {not_existing_class}")
print(f"Classes that do not have enough images: {not_enough_classes}")

not_enough_classes_list = list(not_enough_classes.keys())
print(f"List of classes to redownload (no dir + not enough imagess): {not_existing_class + not_enough_classes_list}")
print(f"Total number of classes to redownload: {len(not_existing_class + not_enough_classes_list)} / {len(target_class)}")