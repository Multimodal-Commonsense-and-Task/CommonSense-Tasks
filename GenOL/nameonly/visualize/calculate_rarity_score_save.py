# High RMD가 High rarity를 갖고, Low RMD가 Low rarity를 갖는다는 것을 위해 qualitative figure를 만들 때 필요한 것들
# Rarity score를 구해서 저장해야 함. PACS 대상으로.

import os
import random
import numpy as np
import natsort
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rarity.rarity_score import *
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import models, transforms
from utils import get_topk_average
from classes import *

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
use_sample_size = False
# generated_base_path = './datasets/ablations/pacs_base'
generated_full_path = './datasets/ablations/pacs_full'
# generated_rmd_path = './datasets/PACS/final/PACS_final_ensembled_RMD_classwise_temp_1'
classes = os.listdir(generated_full_path)
classes.remove('person')

full_score_list_all_class = []
class_sample_rarity_dict = {}
for cls in classes:
    num_samples = PACS_count[cls]
    real_cls_path = os.path.join(real_path, cls)
    generated_full_cls_path = os.path.join(generated_full_path, cls)
    full_score_dict = calculate_rarity_score(real_cls_path, generated_full_cls_path, use_same_size=use_sample_size)
    
    scores = []
    for path, score in full_score_dict.items():
        scores.append((path, score))
    
    class_sample_rarity_dict[cls] = scores

with open('./RMD_scores/Rarity_scores_PACS.pkl', 'wb') as f:
    pickle.dump(class_sample_rarity_dict, f)

breakpoint()
# Plot full score figure (average across all classes)
base_score_list_average_across_class = []
full_score_list_average_across_class = []
rmd_score_list_average_across_class = []
for i in range(len(p_values)):
    base_score_all_class = [base_score_list_all_class[j][i] for j in range(len(classes))]
    full_score_all_class = [full_score_list_all_class[j][i] for j in range(len(classes))]
    rmd_score_all_class = [rmd_score_list_all_class[j][i] for j in range(len(classes))]
    # base_score_all_class = [score for score in base_score_all_class if score > 0.0] # Remove zero scores (person class)
    # full_score_all_class = [score for score in full_score_all_class if score > 0.0] # Remove zero scores (person class)
    # rmd_score_all_class = [score for score in rmd_score_all_class if score > 0.0] # Remove zero scores (person class)
    base_score_list_average_across_class.append(sum(base_score_all_class) / len(base_score_all_class))
    full_score_list_average_across_class.append(sum(full_score_all_class) / len(full_score_all_class))
    rmd_score_list_average_across_class.append(sum(rmd_score_all_class) / len(rmd_score_all_class))




# Add labels, title, legend
plt.xlabel('p(%)')
plt.ylabel("Average Score")
plt.title(f"Average rarity score for top p% samples (Average)")

plt.legend()
plt.savefig(f"./vis/average.png")