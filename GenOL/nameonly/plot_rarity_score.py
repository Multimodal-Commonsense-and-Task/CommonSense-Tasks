import os
import random
import numpy as np
import natsort
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rarity.rarity_score import *
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import models, transforms
from utils import get_topk_average
from classes import *
import pickle

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image)
    return image_tensor.unsqueeze(0)

def calculate_features_CLIP(image_paths, model, processor, device, normalize=True):
    # Preprocess the images
    image_features = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            image = Image.open(image_path)
        except Exception as e:
            # append a dummy feature
            print(f"Error in opening {image_path}: {e}, appending a dummy feature")
            image_features.append(torch.zeros(1, 512).to(device))
            continue
        with torch.no_grad():
            image_input = processor(images=image, return_tensors="pt").to(device)
            image_feature = model.get_image_features(**image_input) # [1, 512]
            image_features.append(image_feature)

    # Normalize the features
    image_features = torch.cat(image_features, dim=0)
    if normalize:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features

def calculate_features_vgg(image_paths, device, normalize=True):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    model.classifier = model.classifier[:-1]
    
    model.eval()
    image_features = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            image = preprocess_image(image_path).to(device)
        except Exception as e:
            # append a dummy feature
            print(f"Error in opening {image_path}: {e}, appending a dummy feature")
            image_features.append(torch.zeros(1, 4096).to(device))
            continue
        with torch.no_grad():
            image_feature = model(image)
            image_features.append(image_feature)
    
    # Normalize the features
    image_features = torch.cat(image_features, dim=0)
    if normalize:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features

def calculate_rarity_score(real_path, generated_path, nearest_k=3, use_same_size=True, sample_size=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    images_real = [os.path.join(real_path, image) for image in os.listdir(real_path)]
    images_generated = [os.path.join(generated_path, image) for image in os.listdir(generated_path)]
    if sample_size:
        print(f"Set sample size to {sample_size}!")
        random.shuffle(images_real); images_real = images_real[:sample_size]
        random.shuffle(images_generated); images_generated = images_generated[:sample_size]
    elif use_same_size:
        sample_size = min(len(images_real), len(images_generated))
        print(f"Sample size is automatically adjusted to {sample_size}!")
        random.shuffle(images_real); images_real = images_real[:sample_size]
        random.shuffle(images_generated); images_generated = images_generated[:sample_size]
    # real_features = calculate_features_CLIP(images_real, model, processor, device, normalize=False)
    # generated_features = calculate_features_CLIP(images_generated, model, processor, device, normalize=False)[:189]
    real_features = calculate_features_vgg(images_real, device, normalize=False)
    generated_features = calculate_features_vgg(images_generated, device, normalize=False)
    
    manifold = MANIFOLD(real_features=real_features, fake_features=generated_features)
    score, score_index = manifold.rarity(k=nearest_k)

    # Return item to score dictionary
    score_dict = {k:v for k, v in zip(images_generated, score)}
    
    return score_dict
    
sample_count_dict = PACS_count
real_path = './datasets/ablations/pacs_ma'
generated_base_path = './datasets/ablations/pacs_base'
generated_full_path = './datasets/ablations/pacs_full'
# generated_rmd_path = './datasets/PACS/final/PACS_final_ensembled_RMD_classwise_temp_1'
classes = os.listdir(real_path)
classes.remove('person')

topk_ratio = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
p_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
base_score_list_all_class = []
full_score_list_all_class = []
# rmd_score_list_all_class = []
for cls in classes:
    print(f"Processing {cls}...")
    num_samples = PACS_count[cls]
    real_cls_path = os.path.join(real_path, cls)
    generated_base_cls_path = os.path.join(generated_base_path, cls)
    generated_full_cls_path = os.path.join(generated_full_path, cls)
    # generated_rmd_cls_path = os.path.join(generated_rmd_path, cls)
    base_score_dict = calculate_rarity_score(real_cls_path, generated_base_cls_path, sample_size=num_samples)
    full_score_dict = calculate_rarity_score(real_cls_path, generated_full_cls_path, sample_size=num_samples)
    # rmd_score_dict = calculate_rarity_score(real_cls_path, generated_rmd_cls_path, sample_size=num_samples)
    base_score_list = list(base_score_dict.values()); full_score_list = list(full_score_dict.values())
    # rmd_score_list = list(rmd_score_dict.values())
    
    # Plot top-k ratio iteratively
    topk_nums = [int(len(full_score_list) * ratio) for ratio in topk_ratio]
    base_topk_average_list = []; full_topk_average_list = []; rmd_topk_average_list = []
    for k in topk_nums:
        base_topk_average_list.append(get_topk_average(base_score_list, k))
        full_topk_average_list.append(get_topk_average(full_score_list, k))
        # rmd_topk_average_list.append(get_topk_average(rmd_score_list, k))
    base_score_list_all_class.append(base_topk_average_list)
    full_score_list_all_class.append(full_topk_average_list)
    # rmd_score_list_all_class.append(rmd_topk_average_list)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=False)
    # plt.figure()
    ax.plot(p_values, base_topk_average_list, '-o', label='PACS Base Prompt', color='lightsteelblue')
    ax.plot(p_values, full_topk_average_list, '-o', label='PACS Prompt Rewrites', color='indigo')
    # plt.plot(p_values, rmd_topk_average_list, '-o', label='PACS RMD Ensembled')
    
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    
    # Add labels, title, legend
    ax.set_xlabel('p(%)', fontsize=20)
    ax.set_ylabel("Average Score", fontsize=20)
    ax.set_title(f"Average Rarity score for top p% samples ({cls})", fontsize=24, pad=10)
    
    # Set legend
    handle, label = ax.get_legend_handles_labels()
    fig.legend(handle, label, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=20)
    plt.savefig(f"./vis/{cls}.png")
    plt.savefig(f"./vis/{cls}.pdf", bbox_inches="tight")
    

# Plot full score figure (average across all classes)
base_score_list_average_across_class = []
full_score_list_average_across_class = []
rmd_score_list_average_across_class = []
for i in range(len(p_values)):
    base_score_all_class = [base_score_list_all_class[j][i] for j in range(len(classes))]
    full_score_all_class = [full_score_list_all_class[j][i] for j in range(len(classes))]
    # rmd_score_all_class = [rmd_score_list_all_class[j][i] for j in range(len(classes))]
    # base_score_all_class = [score for score in base_score_all_class if score > 0.0] # Remove zero scores (person class)
    # full_score_all_class = [score for score in full_score_all_class if score > 0.0] # Remove zero scores (person class)
    # rmd_score_all_class = [score for score in rmd_score_all_class if score > 0.0] # Remove zero scores (person class)
    base_score_list_average_across_class.append(sum(base_score_all_class) / len(base_score_all_class))
    full_score_list_average_across_class.append(sum(full_score_all_class) / len(full_score_all_class))
    # rmd_score_list_average_across_class.append(sum(rmd_score_all_class) / len(rmd_score_all_class))

cls_score_dict = {}
cls_score_dict['class'] = {}
for i, cls in enumerate(classes):
    base_scores = base_score_list_all_class[i]; full_scores = full_score_list_all_class[i]
    cls_score_dict['class'][cls] = {'base':base_scores, 'full':full_scores}
cls_score_dict['average'] = {'base': base_score_list_average_across_class, 'full': full_score_list_average_across_class}

with open('RMD_scores/Rarity_scores_PACS_ablation_plot.pkl', 'wb') as f:
    pickle.dump(cls_score_dict, f)


# base_score_list_average_across_class = [39.994372608454356, 38.71379669635387, 38.24212025096372, 37.71779187736838, 37.143611423522714, 36.677897895765284, 36.254034519580806, 35.939336965859404, 35.57268412874936, 35.33427674337708]
# full_score_list_average_across_class = [40.72737399435413, 40.25892656452103, 39.782478501131756, 39.183109830547586, 38.70449629247954, 38.37997598218984, 38.024100874821066, 37.78082105806237, 37.439336340793936, 37.191728857687174]
# rmd_score_list_average_across_class = [40.54305742052448, 39.873611783428764, 39.44144410267957, 38.94742464851221, 38.36688156506069, 38.023469494969056, 37.60784060107608, 37.130564651558885, 36.29495883310485, 35.837643557388134]

fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=False)
ax.plot(p_values, base_score_list_average_across_class, '-o', label='PACS Base Prompt', color='lightsteelblue')
ax.plot(p_values, full_score_list_average_across_class, '-o', label='PACS Prompt Rewrites', color='indigo')
# ax.plot(p_values, rmd_score_list_average_across_class, '-o', label='PACS RMD Ensembled')

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)

# Add labels, title, legend
ax.set_xlabel('p(%)', fontsize=20)
ax.set_ylabel("Average Score", fontsize=20)
ax.set_title(f"Average rarity score for top p% samples (Average)", fontsize=24, pad=10)

handle, label = ax.get_legend_handles_labels()
fig.legend(handle, label, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=20)
plt.savefig(f"./vis/average.png")
plt.savefig(f"./vis/average.pdf", bbox_inches="tight")