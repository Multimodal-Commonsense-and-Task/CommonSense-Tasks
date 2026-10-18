import os, random, shutil
import pickle
import numpy as np
from classes import *
from tqdm import tqdm

# 일단 RMD score를 다 구하고, aggregate해서 model마다 평균을 구하고, 그걸 바탕으로 model마다 probability를 구한다.
# 그래서 stage 1에서는 model을 뽑고, 다음으로 model 내에서는 samplewise RMD score로 뽑는 것이다.

def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

Equalweight = False
TopK = False
BottomK = False

INVERSE = False
TEMPERATURE = 3
count_dict = PACS_count
rmd_pickle_path = './RMD_scores/PACS_final.pkl'

# PACS
target_path = './datasets/neurips/PACS/final/PACS_final_twostage_temp_3'

# PATH_dict = {
#         'flickr': './datasets/PACS/backup/PACS_flickr_for_RMD',
#         'google': './datasets/PACS/backup/PACS_google_for_RMD',
#         'bing': './datasets/PACS/backup/PACS_bing_for_RMD',
# }
PATH_dict = {
    'sdxl': './datasets/neurips/PACS/PACS_sdxl_diversified_subsampled',
    'dalle2': './datasets/neurips/PACS/PACS_dalle2_subsampled',
    'floyd': './datasets/neurips/PACS/PACS_floyd_subsampled',
    'cogview2': './datasets/neurips/PACS/PACS_cogview2_subsampled',
}

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
    # First convert the mapping
    # Original(RMD_scores): {(model, cls): [(image_path, RMD_score), ...]}
    # New: {model: RMD_score, ...}

    # For top-k
    image_rmd_scores = []
    for (model, cls_), images in RMD_scores.items():
        if cls_ == cls:
            for image, score in images:
                image_rmd_scores.append((model, image, score))
    
    # Get sample_path -> score mapping
    sample_model_RMD_mapping = {}; model_sample_RMD_mapping = {model: [] for model in PATH_dict.keys()}
    for sample in image_rmd_scores:
        model_sample_RMD_mapping[sample[0]].append((sample[1], sample[2])) # sample_path, score
        sample_model_RMD_mapping[sample[1]] = sample[0], sample[2] # model, score

    # Get average score of each model
    model_scores_dict = {}
    for model, sample_path_scores in model_sample_RMD_mapping.items():
        scores_list = [path_score[1] for path_score in sample_path_scores]
        model_scores_dict[model] = np.mean(scores_list)
        print(f"Class {cls}, model {model}, average RMD score: {model_scores_dict[model]}")
    
    if 'dalle2' in model_scores_dict and cls == 'underwear':
        model_scores_dict['dalle2'] = 0.01 # To avoid numerical instability
    
    while True:
        model_scores = np.array([score for score in model_scores_dict.values()])
        model_probabilities = softmax_with_temperature(model_scores, TEMPERATURE)
        
        if INVERSE:
            # To get the inverse probabilities, first handle the numerical instability
            if np.min(model_probabilities) < 0:
                model_probabilities -= np.min(model_probabilities)
            # Handle devision by zero
            if np.sum(model_probabilities) == 0:
                raise ValueError("All probabilities are zero")
            model_probabilities = 1 / model_probabilities
            model_probabilities /= np.sum(model_probabilities) # Normalize the probabilities

        chosen_model = random.choices(list(model_scores_dict.keys()), weights=model_probabilities, k=1)[0]

        # 선택된 model 내에서도 RMD score based로 sample을 선택 (원래는 랜덤하게 했음)
        sample_scores_list = model_sample_RMD_mapping[chosen_model]
        sample_paths = [sample_score[0] for sample_score in sample_scores_list]
        sample_scores = [sample_score[1] for sample_score in sample_scores_list]
        sample_probabilities = softmax_with_temperature(sample_scores, TEMPERATURE)
        chosen_sample = random.choices(sample_paths, weights=sample_probabilities, k=1)
        sample_scores_list = [tuple for tuple in sample_scores_list if tuple[0] != chosen_sample[0]] # Remove selected sample
        model_sample_RMD_mapping[chosen_model] = sample_scores_list

        # Update ensembled_images
        ensembled_images[cls].append({'model': chosen_model, 'image': chosen_sample[0]})
        model_class_selected_counter[chosen_model][cls] += 1

        # Break if full
        if len(ensembled_images[cls]) == count_dict[cls]:
                print(f"Break for class {cls} with {len(ensembled_images[cls])} images")
                break

# Check the number of images selected for each model, for each class
for model, class_counter in model_class_selected_counter.items():
    print(f"Model {model} selected for each class:")
    print(class_counter)

# Sanity check the number of images for each class
for cls, images in ensembled_images.items():
    if len(images) != count_dict[cls]:
        raise ValueError(f"Class {cls} has {count_dict[cls]} images but {len(images)} images are selected")

# Copy all the images to the target path
# Remove target path if already exists
if os.path.exists(target_path):
    print(f"[WARNING]: Removing already existing target path - {target_path}")
    shutil.rmtree(target_path)

for cls, images in ensembled_images.items():
    image_counter = 0
    target_cls_path = os.path.join(target_path, cls)
    os.makedirs(target_cls_path, exist_ok=True)
    for img in images:
        model = img['model']; image_name = img['image']
        new_image_name = f"{model}_{str(image_counter).zfill(6)}.png"
        image_counter += 1
        shutil.copy(image_name, os.path.join(target_cls_path, new_image_name))