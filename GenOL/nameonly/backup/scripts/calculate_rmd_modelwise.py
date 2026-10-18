import torch
import os
import natsort
import numpy as np
import pickle
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
from classes import *

def calculate_features(image_paths, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

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
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

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
    classes = sorted(list(set(class_labels)))
    mu_specific = {}
    cov_specific = []

    for cls in classes:
        cls_features = image_features[class_labels == cls]
        mu_specific[cls] = cls_features.mean(axis=0)
        cov_specific.append((cls_features - mu_specific[cls]).T @ (cls_features - mu_specific[cls]))

    # cov_specific should be averaged over all classes
    cov_specific = np.array(cov_specific).sum(axis=0) / len(image_features)

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

def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

if __name__ == "__main__":
    count_dict = pacs_count
    image_paths = {
        'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl',
        'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_dalle2',
        'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_floyd',
        'cogview': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_cogview2'
    }
    classes = sorted(list(count_dict.keys()))
    images_dict = {model: {cls: [] for cls in classes} for model in image_paths.keys()}
    for model, path in image_paths.items():
        for cls in classes:
            cls_path = os.path.join(path, cls)
            images = natsort.natsorted(os.listdir(cls_path))
            images = [os.path.join(cls_path, image) for image in images]
            images_dict[model][cls] = images
    
    # Adjust the number of images to be the same for each model
    results = {cls: [] for cls in classes}
    for cls in classes:
        for model in image_paths.keys():
            results[cls].append(len(images_dict[model][cls]))
    min_dict = {cls: min(results[cls]) for cls in classes}   

    # Remove the extra images
    for cls in classes:
        for model in image_paths.keys():
            images_dict[model][cls] = images_dict[model][cls][:min_dict[cls]]


            
    # Concat all images into one list and generate corresponding class lables
    concatenated_images = []
    class_labels = []
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
    features = calculate_features(concatenated_images, model)
    mu_agnostic, cov_agnostic = compute_class_agnostic_params(features)
    mu_specific_dict, cov_specific = compute_class_specific_params(features, class_labels) # (C, 512), (C, 512, 512)
    
    # # Calcuate the mahalanobis_specific distance
    # distance_specific = mahalanobis_distance_specific(features, mu_specific_dict, cov_specific, class_labels)
    # distance_agnostic = mahalanobis_distance_agnostic(features, mu_agnostic, cov_agnostic)

    # RMD = distance_specific - distance_agnostic
    RMD = mahalanobis_distance_manually(features, mu_agnostic, cov_agnostic, mu_specific_dict, cov_specific, class_labels)

    # Split RMD scores using model labels
    RMD_each_model = {}
    for model in image_paths.keys():
        RMD_each_model[model] = RMD[model_labels == model]
    
    # Generate 
    

    # Print the top 5 and bottom 5 RMD scores of each model
    concatenated_images = np.array(concatenated_images)
    for model in image_paths.keys():
        top5_indices = RMD_each_model[model].argsort()[-5:]
        bottom5_indices = RMD_each_model[model].argsort()[:5]
        top5_image_paths = concatenated_images[top5_indices]
        bottom5_image_paths = concatenated_images[bottom5_indices]
        print(f"Top 5 RMD for {model}:")
        print(f"{top5_image_paths}, scores: {RMD_each_model[model][top5_indices]}")
        print(f"Bottom 5 RMD for {model}:")
        print(f"{bottom5_image_paths}, scores: {RMD_each_model[model][bottom5_indices]}")


    # Get average RMD scores for each model
    for model in image_paths.keys():
        print(f"Average RMD for {model}: {RMD_each_model[model].mean()}")

    model_RMD_scores = [RMD_each_model[model].mean() for model in image_paths.keys()]
    # Calculate probabilites of numpy array using logit
    temperature_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for temperature in temperature_list:
        probabilities = softmax_with_temperature(model_RMD_scores, temperature)
        # Print {model: probability}
        print(f"Probabilities for temperature {temperature}:")
        for i, model in enumerate(image_paths.keys()):
            print(f"{model}: {probabilities[i]}")