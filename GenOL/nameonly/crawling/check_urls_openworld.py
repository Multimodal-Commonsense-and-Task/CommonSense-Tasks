import os
import json
from classes import *

target_path = "urls/Bongard_flickr"
url_threshold = 30


# Process jsonl file
data_list = []
with open('../generate_twostage/train.jsonl', 'r') as f:
    for line in f:
        json_object = json.loads(line)
        data_list.append(json_object)

# Read all txt files and check the number of URLs
not_existing_class = []
not_enough_classes = {}
for data in data_list:
    uid = data['uid']
    uid_path = os.path.join(target_path, f"{uid}.txt")
    if not os.path.exists(uid_path):
        not_existing_class.append(uid)
        continue
    else:
        with open(uid_path, 'r') as f:
            num_urls = len(f.readlines())
        
        if num_urls < url_threshold:
            print(f"WARNING: Class {uid} has {num_urls} URLs")
            not_enough_classes[uid] = num_urls
    
print(f"Classes that do not have a txt file: {not_existing_class}")
print(f"Classes that do not have enough URLs: {not_enough_classes}")

not_enough_classes_list = list(not_enough_classes.keys())
print(f"List of classes to regenerate (no txt + not enough URLs): {not_existing_class + not_enough_classes_list}")
print(f"Total number of classes to regenerate: {len(not_existing_class + not_enough_classes_list)} / {len(data_list)}")