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

def softmax_with_temperature(z, T): 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

def normalized_RMD_scores(features, class_labels, model_labels):
    # Calculate normalized RMD scores
    # features: (N, 512) numpy array, each row is a normalized feature vector
    # class_labels: (N,) numpy array, each element is the class label of the corresponding feature vector
    # model_labels: (N,) numpy array, each element is the model label of the corresponding feature vector

    # 1. Calculate the class-specific parameters
    # 1-1. Calculate the list of mu_k (k = 1, 2, ..., C), which denotes the mean of the features of each class List[(512,), (512,), ...]
    classes = sorted(list(set(class_labels))) # Get the list of classes (C)
    mu_k_list = []
    for cls in classes:
        mu_k = features[class_labels == cls].mean(axis=0)
        mu_k_list.append(mu_k)
    
    # 1-2. Calculate the class-wise covariance matrix of the features
    #  The covariance matrix is averaged over all classes to avoid under-fitting
    print(f"Calculating the class-wise covariance matrix for {len(classes)} classes")
    cov_k_list = []
    for i, cls in enumerate(tqdm(classes)):
        difference = features[class_labels == cls] - mu_k_list[i] # (N_k, 512)
        inner_loop_list = []
        for j in range(len(difference)):
            inner_loop_list.append(difference[j].reshape(-1, 1) @ difference[j].reshape(1, -1))
        cov_k = np.array(inner_loop_list).sum(axis=0) # (512, 512)
        cov_k_list.append(cov_k)
    
    cov = np.array(cov_k_list).sum(axis=0) / len(features)
    
    # 2. Calculate the class-agnotic parameters
    print(f"Calculating the class-agnostic parameters")
    mu_agn = features.mean(axis=0) # (512,)
    difference = features - mu_agn # (N, 512)
    inner_loop_list = []

    # Killed after this line
    print(f"Calculating the class-agnostic covariance matrix for {len(features)} samples")
    cov_agn = np.einsum('ij,ik->jk', difference, difference) / len(features)

    # 3. Calculate the class_specific distance M(x_i, y_i)
    print(f"Calculating inverse of the class-wise covariance matrix for {len(classes)} classes")
    inverse_cov = np.linalg.inv(cov)
    
    # 3-1. Initialize the N x C matrix to store M(x_i, y_i)
    print(f"Calculating M(x_i, y_i) for {len(features)} samples and {len(classes)} classes")
    M = np.zeros((len(features), len(classes)))
    for i in tqdm(range(len(features))):
        for j in range(len(classes)):
            difference = features[i] - mu_k_list[j] # (512,)
            M[i, j] = -1.0 * difference @ inverse_cov @ difference.T
    
    # 4. Calculate the class-agnostic distance M_{agn}(x_i)
    print(f"Calculating M_agn(x_i) for {len(features)} samples")
    inverse_cov_agn = np.linalg.inv(cov_agn)
    M_agn = np.zeros((len(features)))
    for i in range(len(features)):
        difference = features[i] - mu_agn
        M_agn[i] = -1.0 * difference @ inverse_cov_agn @ difference.T
    
    # 5. Calculate the RMD scores: RMD(x_i, y_i) = M(x_i, y_i) - M_{agn}(x_i)
    RMD = M - M_agn.reshape(-1, 1)

    # 6. Return the samplewise RMD scores using class labels ([N, C] to [N,])
    RMD_samplewise = np.zeros((len(features)))
    for i in range(len(features)):
        RMD_samplewise[i] = RMD[i, classes.index(class_labels[i])]
    
    # 7. Normalize the RMD scores using the class labels
    # 7-1. Calculate the mean and standard deviation of the RMD scores for each class
    mean_dict = {cls: RMD_samplewise[class_labels == cls].mean() for cls in classes}
    std_dict = {cls: RMD_samplewise[class_labels == cls].std() for cls in classes}

    # 7-2. Normalize the RMD scores (each class has zero mean and unit variance)
    RMD_normalized = np.zeros((len(features)))
    for i in range(len(features)):
        cls = class_labels[i]
        RMD_normalized[i] = (RMD_samplewise[i] - mean_dict[cls]) / std_dict[cls]
    
    return RMD_normalized

if __name__ == "__main__":
    samplewise = True
    count_dict = pacs_count
    rmd_pickle_path = 'RMD_scores_normalized_PACS.pkl'
    image_paths = {
        'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl_diversified',
        'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_dalle2',
        'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_floyd',
        'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_cogview2',
        # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_kandinsky'
    }
    # image_paths = {
    #     'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_sdxl_diversified',
    #     'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_dalle2',
    #     'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_floyd',
    #     'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/DomainNet/DomainNet_cogview2',
    #     # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_kandinsky'
    # }
    # image_paths = {
    #     'sdxl': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_sdxl_diversified',
    #     'dalle2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_dalle2',
    #     'floyd': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_floyd',
    #     'cogview2': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_cogview2',
    #     # 'kandinsky': '/workspace/home/user/seongwon/crawling/crawler/datasets/cct/cct_kandinsky'
    # }
    classes = sorted(list(count_dict.keys()))
    images_dict = {model: {cls: [] for cls in classes} for model in image_paths.keys()}
    for model, path in image_paths.items():
        for cls in classes:
            cls_path = os.path.join(path, cls)
            images = natsort.natsorted(os.listdir(cls_path))
            images = [os.path.join(cls_path, image) for image in images]
            images_dict[model][cls] = images
    
    # Adjust the number of images to be the same for each model
    if not samplewise:
        results = {cls: [] for cls in classes}
        for cls in classes:
            for model in image_paths.keys():
                results[cls].append(len(images_dict[model][cls]))
        
        print(f"Number of images for each class for each model:")
        print(results)

        min_dict = {cls: min(results[cls]) for cls in classes} # Number of minimum images for each class  
        print("Minimum number of images for each class:")
        print(min_dict)

    # for cls in classes:
    #     # If the number of images is less than the minimum, set the minimum to the number of images
    #     if min_dict[cls] < count_dict[cls]:
    #         min_dict[cls] = count_dict[cls]
    
    if not samplewise:
        print("Minimum number of images for each class after adjustment:")
        print(min_dict)
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
    features = calculate_features(concatenated_images, model) # (N, 512)

    RMD = normalized_RMD_scores(features, class_labels=class_labels, model_labels=model_labels)

    # Split RMD scores using model labels
    RMD_each_model = {}
    for model in image_paths.keys():
        RMD_each_model[model] = RMD[model_labels == model]
    
    
    # Generate model - class - image_path - RMD dictionary
    model_class_image_RMD = {}
    for model in image_paths.keys():
        for cls in classes:
            indices = np.where((model_labels == model) & (class_labels == cls))[0]
            RMDs = RMD[indices]
            paths = np.array(concatenated_images)[indices]
            model_class_image_RMD[(model, cls)] = list(zip(paths, RMDs))

    # Save the RMD scores as pickle
    with open(rmd_pickle_path, 'wb') as f:
        pickle.dump(model_class_image_RMD, f)

    # Print the top 5 and bottom 5 RMD scores of each model
    concatenated_images = np.array(concatenated_images)
    for model in image_paths.keys():
        top5_indices = RMD_each_model[model].argsort()[-100:]
        bottom5_indices = RMD_each_model[model].argsort()[:100]
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