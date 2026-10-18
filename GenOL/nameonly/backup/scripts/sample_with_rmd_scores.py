import os, random, shutil
import pickle
import numpy as np
from classes import *
from utils import softmax_with_temperature

TEMPERATURE = 3
count_dict = cct_count

target_path = '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_RMD_final_samplewise_temp_3'
pickle_path = '/workspace/home/user/seongwon/crawling/crawler/RMD_scores.pkl'

PATH_dict = {
    'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_sdxl_diversified',
    'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_dalle2',
    'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_floyd',
    'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_cogview',
    # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_kandinsky'
}

# Load the RMD scores
with open(pickle_path, 'rb') as f:
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
model_selected_counter = {model: 0 for model in PATH_dict.keys()}

for cls in count_dict.keys():
    # First convert the mapping
    # Original(RMD_scores): {(model, cls): [(image_path, RMD_score), ...]}
    # New: {cls: [(model, image_path, RMD_score), ...]}
    class_to_images = {cls_: [] for cls_ in count_dict.keys()}
    for (model, cls_), images in RMD_scores.items():
        if cls_ == cls:
            for image, score in images:
                class_to_images[cls].append((model, image, score))

    # Randomly shuffle the images
    for model in PATH_dict.keys():
        random.shuffle(class_to_images[cls])
    
    # Randomly choose the model according to the ensemble probability
    while True:
        scores = np.array([score for _, _, score in class_to_images[cls]])
        if len(scores) == 0:
            breakpoint()
        probabilities = softmax_with_temperature(scores, TEMPERATURE)
        chosen_sample = random.choices(class_to_images[cls], weights=probabilities, k=1)[0]
        chosen_model, chosen_image, _ = chosen_sample

        if len(class_to_images[cls]) > 0:
            ensembled_images[cls].append({'model': chosen_model, 'image': chosen_image})
            class_to_images[cls].remove(chosen_sample)
            model_selected_counter[chosen_model] += 1
        if len(ensembled_images[cls]) == count_dict[cls]:
            print(f"Break for class {cls} with {len(ensembled_images[cls])} images")
            break

# Check the number of images selected for each model
for model, count in model_selected_counter.items():
    print(f"Model {model} selected {count} times")

# Sanity check the number of images for each class
for cls, images in ensembled_images.items():
    if len(images) != count_dict[cls]:
        raise ValueError(f"Class {cls} has {count_dict[cls]} images but {len(images)} images are selected")

# Copy all the images to the target path
for cls, images in ensembled_images.items():
    image_counter = 0
    target_cls_path = os.path.join(target_path, cls)
    os.makedirs(target_cls_path, exist_ok=True)
    for img in images:
        model = img['model']; image_name = img['image']
        new_image_name = f"{model}_{str(image_counter).zfill(6)}.png"
        image_counter += 1
        shutil.copy(image_name, os.path.join(target_cls_path, new_image_name))