import os, random, shutil
from classes import *

count_dict = cct_count

target_path = '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_equalweighted'

PATH_dict = {
    'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_sdxl_diversified',
    'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_dalle2',
    'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_floyd',
    'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_cogview2',
    # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/pacs_kandinsky'
}

# PATH_dict = {
#     'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl_diversified',
#     'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_dalle2',
#     'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_floyd',
#     'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_cogview2',
#     # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/pacs_kandinsky'
# }

ensemble_probability_dict = {
    'sdxl': 0.25,
    'dalle2': 0.25,
    'floyd': 0.25,
    'cogview2': 0.25,
    # 'kandinsky': 0.2321615603261025
}

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
    model_images_dict = {model: model_to_images_dict[model][cls] for model in PATH_dict.keys()}
    
    # Randomly choose the model according to the ensemble probability
    while True:
        models_list = list(PATH_dict.keys())
        probabilites = list(ensemble_probability_dict.values())
        chosen_model = random.choices(models_list, weights=probabilites, k=1)[0]
        model_selected_counter[chosen_model] += 1
        if len(model_images_dict[chosen_model]) > 0:
            chosen_image = model_images_dict[chosen_model].pop()
            chosen_image_path = os.path.join(PATH_dict[chosen_model], cls, chosen_image)
            ensembled_images[cls].append({'model': chosen_model, 'image': chosen_image_path})
        if len(ensembled_images[cls]) == count_dict[cls]:
            # print(f"Break for class {cls} with {len(ensembled_images[cls])} images")
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