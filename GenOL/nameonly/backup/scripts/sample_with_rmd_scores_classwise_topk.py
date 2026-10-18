import os, random, shutil
import pickle
import numpy as np
from classes import *
from utils import softmax_with_temperature
from tqdm import tqdm

TopK = False # If false, BottomK
count_dict = pacs_count
rmd_pickle_path = './RMD_scores_PACS.pkl'

# PACS
target_path = '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_final_bottomk'

# CCT
# target_path = '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_RMD_final_classwise_temp_3'

# DomainNet
# target_path = '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_RMD_final_classwise_temp_1'

PATH_dict = {
    'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl_diversified',
    'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_dalle2',
    'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_floyd',
    'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_cogview2',
    # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_kandinsky'
}
# PATH_dict = {
#     'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_sdxl_diversified',
#     'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_dalle2',
#     'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_floyd',
#     'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_cogview',
#     # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_kandinsky'
# }
# PATH_dict = {
#     'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_sdxl_diversified',
#     'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_dalle2',
#     'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_floyd',
#     'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_cogview2',
#     # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_kandinsky'
# }

# Load the RMD scores
with open(rmd_pickle_path, 'rb') as f:
    RMD_scores = pickle.load(f)

# Load the images
model_to_images_dict = {}
for model, path in PATH_dict.items():
    model_to_images_dict[model] = {cls: os.listdir(os.path.join(path, cls)) for cls in count_dict.keys()}

# Shuffle the images
for model in PATH_dict.keys():
    for cls in count_dict.keys():
        random.shuffle(model_to_images_dict[model][cls])

ensembled_images = {cls: [] for cls in count_dict.keys()}
model_class_selected_counter = {model: {cls: 0 for cls in count_dict.keys()} for model in PATH_dict.keys()}

for cls in tqdm(list(count_dict.keys())):
    # First get the list of paths / rmd scores of specific class
    # [(model, image_path, RMD_score), ...]
    image_rmd_scores = []
    for (model, cls_), images in RMD_scores.items():
        if cls_ == cls:
            for image, score in images:
                image_rmd_scores.append((model, image, score))
    
    # Sort the list by RMD score
    if TopK:
        image_rmd_scores.sort(key=lambda x: x[2], reverse=True)
    else:
        image_rmd_scores.sort(key=lambda x: x[2], reverse=False)
    
    # Assign the top images to the ensembled_images
    ensembled_images[cls] = [image_rmd_scores[i] for i in range(count_dict[cls])]
    for model, image, score in ensembled_images[cls]:
        model_class_selected_counter[model][cls] += 1

# Check the number of images selected for each model, for each class
for model, class_counter in model_class_selected_counter.items():
    print(f"Model {model} selected for each class:")
    print(class_counter)

# Sanity check the number of images for each class
for cls, images in ensembled_images.items():
    if len(images) != count_dict[cls]:
        raise ValueError(f"Class {cls} has {count_dict[cls]} images but {len(images)} images are selected")

# Copy all the images to the target path
for cls, images in ensembled_images.items():
    image_counter = 0
    target_cls_path = os.path.join(target_path, cls)
    os.makedirs(target_cls_path, exist_ok=True)
    for model, image, _ in images:
        new_image_name = f"{model}_{str(image_counter).zfill(6)}.png"
        image_counter += 1
        shutil.copy(image, os.path.join(target_cls_path, new_image_name))