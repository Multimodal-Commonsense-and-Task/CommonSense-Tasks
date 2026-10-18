import torch
import yaml
import os
import natsort
import random
import torch.nn as nn
import numpy as np
import json
import argparse
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from torchvision import transforms as transforms
from PIL import Image
from tqdm import tqdm
from classes import *
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from utils import ImageDataset
from torch.utils.data import DataLoader

# For DINO
dino_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet
    # transforms.Normalize((0.490, 0.441, 0.403), (0.263, 0.234, 0.230)), # PACS
    transforms.Normalize((0.481, 0.442, 0.411), (0.252, 0.224, 0.220)) # DomainNet
])

# For DINOv2
dinov2_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet
    # transforms.Normalize(mean=(0.490, 0.441, 0.403), std=(0.263, 0.234, 0.230)), # PACS
])
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)

def calculate_features_dinov2(image_paths, model):
    # dino_vitb16, dino_vitb8, dino_vits16, dino_vits8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    # Preprocess the images
    image_features = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            image = Image.open(image_path)
        except Exception as e:
            # append a dummy feature
            print(f"Error in opening {image_path}: {e}, appending a dummy feature")
            image_features.append(torch.zeros(1, 768).cpu())
            continue
        with torch.no_grad():
            # Get image features
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_feature = outputs.last_hidden_state
            image_feature = image_feature.mean(dim=1).cpu()
            image_features.append(image_feature)

    # Normalize the features
    image_features = torch.cat(image_features, dim=0) # [N, 768]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # dim=-1 -> torch.Size([N, 1]) -> Normalize each data

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features


def calculate_features_dino(image_paths, model):
    # dino_vitb16, dino_vitb8, dino_vits16, dino_vits8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device)
    model.eval()

    # Preprocess the images
    image_features = []
    for i, image_path in enumerate(tqdm(image_paths)):
        try:
            image = Image.open(image_path)
        except Exception as e:
            # append a dummy feature
            print(f"Error in opening {image_path}: {e}, appending a dummy feature")
            image_features.append(torch.zeros(1, 768).cpu())
            continue
        with torch.no_grad():
            # Get image features
            image = dino_transform(image).unsqueeze(0).to(device)
            image_feature = model(image).cpu() # [1, 768]
            image_features.append(image_feature)

    # Normalize the features
    image_features = torch.cat(image_features, dim=0) # [N, 768]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # dim=-1 -> torch.Size([N, 1]) -> Normalize each data

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features

def calculate_features_clip(image_paths, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    dataset = ImageDataset(image_paths, processor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    # Preprocess the images
    image_features = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            pixel_values = batch['pixel_values'].squeeze().to(device)
            image_feature = model.get_image_features(pixel_values=pixel_values)
            image_features.append(image_feature.cpu())
    
    # Normalize the features
    image_features = torch.cat(image_features, dim=0) # [N, 512]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # dim=-1 -> torch.Size([N, 1]) -> Normalize each data

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features

def compute_class_agnostic_params(image_features):
    # mu_agnostic: (512,) / cov_agnostic: (512, 512)
    mu_agnostic = image_features.mean(axis=0)
    cov_agnostic = (image_features - mu_agnostic).T @ (image_features - mu_agnostic)

    cov_agnostic = cov_agnostic / len(image_features)

    return mu_agnostic, cov_agnostic


def compute_class_specific_params(image_features, class_labels):
    # mu_specific: (C, 512) / cov_specific: (C, 512, 512)
    # Image features, class_labels are already aligned
    classes = sorted(list(set(class_labels)))
    mu_specific = {}
    cov_specific = []

    for cls in classes:
        cls_features = image_features[class_labels == cls]
        mu_specific[cls] = cls_features.mean(axis=0)
        cov_specific.append((cls_features - mu_specific[cls]).T @ (cls_features - mu_specific[cls]))

    # cov_specific should be averaged over all classes
    cov_specific = np.array(cov_specific).sum(axis=0) / len(image_features) # (512, 512)

    return mu_specific, cov_specific

def mahalanobis_distance_specific(image_features, mu_specific_dict, sigma_specific, class_labels):
    sigma_specific_inv = np.linalg.inv(sigma_specific)
    mu_specific_list = [mu_specific_dict[cls] for cls in class_labels]
    mu_specific_matrix = np.array(mu_specific_list)

    # Calculate the mahalanobis distance
    diff = image_features - mu_specific_matrix
    distance = -1.0 * diff @ sigma_specific_inv @ diff.T

    return distance

def mahalanobis_distance_agnostic(image_features, mu_agnostic, sigma_agnostic):
    sigma_agnostic_inv = np.linalg.inv(sigma_agnostic)
    diff = image_features - mu_agnostic
    distance = -1.0 * diff @ sigma_agnostic_inv @ diff.T

    return distance

def mahalanobis_distance_manually(image_features, mu_agnostic, sigma_agnostic, mu_specific_dict, sigma_specific, class_labels):
    sigma_specific_inv = np.linalg.inv(sigma_specific)
    sigma_agnostic_inv = np.linalg.inv(sigma_agnostic)
    # Calculate the mahalanobis distance
    RMD = np.zeros((image_features.shape[0]))
    for i in range(image_features.shape[0]):
        feature = image_features[i]; label = class_labels[i]
        difference_specific = feature - mu_specific_dict[label] # (512,)
        M_specific = -1.0 * difference_specific @ sigma_specific_inv @ difference_specific.T

        difference_agnostic = feature - mu_agnostic
        M_agnostic = -1.0 * difference_agnostic @ sigma_agnostic_inv @ difference_agnostic.T

        RMD[i] = M_specific - M_agnostic

    return RMD

def softmax_with_temperature(z, T): 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="yaml path")
    args = parser.parse_args()
    
    # Load yaml file
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    embedding_model = config['embedding_model']
    dataset = config['dataset']
    dataset_dict = count_dict[dataset]
    json_save_path = config['json_save_path']
    image_paths = config['image_paths']
    classes = sorted(list(dataset_dict.keys()))
    images_dict = {model: {cls: [] for cls in classes} for model in image_paths.keys()}

    for model, path in image_paths.items():
        for cls in classes:
            cls_path = os.path.join(path, cls)
            images = natsort.natsorted(os.listdir(cls_path))
            if config['use_ma_size']:
                random.shuffle(images)
                images = images[:dataset_dict[cls]]
                images = natsort.natsorted(images)
            images = [os.path.join(cls_path, image) for image in images]
            images_dict[model][cls] = images

    # Concat all images into one list and generate corresponding class lables
    concatenated_images = []
    class_labels = [] # ['dog', 'dog', 'elephant', 'elephant', 'giraffe', 'giraffe', 'guitar', 'guitar', 'horse', 'horse', 'house', 'house', 'person', 'person']
    model_labels = []
    for model, path in image_paths.items():
        for cls, image_list in images_dict[model].items():
            class_labels += [cls] * len(image_list)
            concatenated_images += image_list
            model_labels += [model] * len(image_list)
    
    # Convert labels to numpy array
    class_labels = np.array(class_labels)
    model_labels = np.array(model_labels)

    # Calculate the features
    if embedding_model == 'clip':
        features = calculate_features_clip(concatenated_images, model)
    elif embedding_model == 'dino':
        features = calculate_features_dino(concatenated_images, model)
    elif embedding_model == 'dinov2':
        features = calculate_features_dinov2(concatenated_images, model)
    else:
        raise ValueError(f"Unknown model: {model}")
    print(f"Calculating class-agnostic parameters...")
    mu_agnostic, cov_agnostic = compute_class_agnostic_params(features)
    print(f"Calculating class-specific parameters...")
    mu_specific_dict, cov_specific = compute_class_specific_params(features, class_labels) # (C, 512), (C, 512, 512)
    
    # # Calcuate the mahalanobis_specific distance
    # distance_specific = mahalanobis_distance_specific(features, mu_specific_dict, cov_specific, class_labels)
    # distance_agnostic = mahalanobis_distance_agnostic(features, mu_agnostic, cov_agnostic)

    # RMD = distance_specific - distance_agnostic
    print(f"Calculating RMD scores...")
    RMD = mahalanobis_distance_manually(features, mu_agnostic, cov_agnostic, mu_specific_dict, cov_specific, class_labels)
    print(f"Done calculating RMD scores...")
    
    # Split RMD scores using model labels
    RMD_each_model = {}
    for model in image_paths.keys():
        RMD_each_model[model] = RMD[model_labels == model]
    
    print(f"Generating model - class - image_path - RMD dictionary...")
    # Generate model - class - image_path - RMD dictionary
    result_dict = {}
    model_class_image_RMD = {}
    for model in image_paths.keys():
        if model not in result_dict:
            result_dict[model] = {}
        for cls in classes:
            if cls not in result_dict[model]:
                result_dict[model][cls] = []
            indices = np.where((model_labels == model) & (class_labels == cls))[0]
            RMDs = RMD[indices]
            paths = np.array(concatenated_images)[indices]
            for i in range(len(RMDs)):
                path = Path(paths[i])
                path = str(Path(*path.parts[-3:]))
                result_dict[model][cls].append({
                    'image_path': path,
                    'score': RMDs[i]
                })

    # Save the RMD scores as json file
    with open(json_save_path, 'w') as f:
        json.dump(result_dict, f)

    # Print the top 5 and bottom 5 RMD scores of each model
    concatenated_images = np.array(concatenated_images)
    for model in image_paths.keys():
        top5_indices = RMD_each_model[model].argsort()[-10:]
        bottom5_indices = RMD_each_model[model].argsort()[:10]
        top5_image_paths = concatenated_images[top5_indices]
        bottom5_image_paths = concatenated_images[bottom5_indices]
        print(f"Top 10 RMD for {model}:")
        print(f"{top5_image_paths}, scores: {RMD_each_model[model][top5_indices]}")
        print(f"Bottom 10 RMD for {model}:")
        print(f"{bottom5_image_paths}, scores: {RMD_each_model[model][bottom5_indices]}")


    # Get average RMD scores for each model
    for model in image_paths.keys():
        print(f"Average RMD for {model}: {RMD_each_model[model].mean()}")

    model_RMD_scores = [RMD_each_model[model].mean() for model in image_paths.keys()]
    # Calculate probabilites of numpy array using logit
    temperature_list = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for temperature in temperature_list:
        probabilities = softmax_with_temperature(model_RMD_scores, temperature)
        # Print {model: probability}
        print(f"Probabilities for temperature {temperature}:")
        for i, model in enumerate(image_paths.keys()):
            print(f"{model}: {probabilities[i]}")
        print()