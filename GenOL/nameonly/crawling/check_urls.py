import os
from classes import *

target_class = food101_count
target_path = f"./urls/food101_flickr"
url_threshold = 480

# Read all txt files and check the number of URLs
not_existing_class = []
not_enough_classes = {}
for cls in target_class:
    cls_path = os.path.join(target_path, f"{cls}.txt")
    if not os.path.exists(cls_path):
        not_existing_class.append(cls)
        continue
    else:
        with open(cls_path, 'r') as f:
            num_urls = len(f.readlines())
        
        if num_urls < url_threshold:
            print(f"WARNING: Class {cls} has {num_urls} URLs")
            not_enough_classes[cls] = num_urls
    
print(f"Classes that do not have a txt file: {not_existing_class}")
print(f"Classes that do not have enough URLs: {not_enough_classes}")

not_enough_classes_list = list(not_enough_classes.keys())
print(f"List of classes to regenerate (no txt + not enough URLs): {not_existing_class + not_enough_classes_list}")
print(f"Total number of classes to regenerate: {len(not_existing_class + not_enough_classes_list)} / {len(target_class)}")