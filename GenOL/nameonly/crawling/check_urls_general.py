import os
import json
from classes import *

target_path = "./urls/bongard_openworld/neg"
url_threshold = 30

txt_list = [os.path.join(target_path, file) for file in os.listdir(target_path)]

# Read all txt files and check the number of URLs
not_existing_class = []
not_enough_classes = {}
for txt_path in txt_list:
    with open(txt_path, 'r') as f:
        num_urls = len(f.readlines())
    
    if num_urls < url_threshold:
        print(f"WARNING: {txt_path} has {num_urls} URLs")

# not_enough_classes_list = list(not_enough_classes.keys())
# print(f"List of classes to regenerate (no txt + not enough URLs): {not_existing_class + not_enough_classes_list}")
# print(f"Total number of classes to regenerate: {len(not_existing_class + not_enough_classes_list)} / {len(data_list)}")