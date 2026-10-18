import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from datetime import datetime
from typing import List
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"

def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

def date_to_unix(date_obj: datetime.date) -> int:
    datetime_obj = datetime(date_obj.year, date_obj.month, date_obj.day)
    return int(time.mktime(datetime_obj.timetuple()))

def get_text_image_similarity(text: str, image_paths: List[str], processor, model) -> List[float]:
    # Load the processor and the model
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Preprocess the text
    with torch.no_grad():
        text_input = processor([text], return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_input) # [1, 512]

    # Preprocess the images
    image_features = []
    dataset = ImageDataset(image_paths, processor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            pixel_values = batch["pixel_values"].squeeze().to(device)
            image_feature = model.get_image_features(pixel_values=pixel_values)
            image_features.append(image_feature)
    
    # Previous code (not using batchwise processing)
    # image_features = []
    # for i, image_path in enumerate(tqdm(image_paths)):
    #     try:
    #         image = Image.open(image_path)
    #     except Exception as e:
    #         # append a dummy feature
    #         print(f"Error in opening {image_path}: {e}, appending a dummy feature")
    #         image_features.append(torch.zeros(1, 512).to(device))
    #         continue
    #     with torch.no_grad():
    #         image_input = processor(images=image, return_tensors="pt").to(device)
    #         image_feature = model.get_image_features(**image_input) # [1, 512]
    #         image_features.append(image_feature)

    # Normalize the features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = torch.cat(image_features, dim=0)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Calculate the similarity
    similarity = (text_features @ image_features.T).squeeze(0)
    return similarity

def distribute_equally(x, y, z, N):
    # Initialize the number of images per type
    images_per_type = [0, 0, 0]
    total_images = x + y + z
    types = ['x', 'y', 'z']
    counts = [x, y, z]

    # Find the number of images to allocate for each type
    for i in range(3):
        images_per_type[i] = min(counts[i], N // 3)

    # Calculate the remaining number of images
    allocated = sum(images_per_type)
    remaining = N - allocated

    # Iterate until all images are allocated
    while remaining > 0 and sum(images_per_type) < total_images:
        # Calculate the number of images left for each type
        leftovers = [counts[i] - images_per_type[i] for i in range(3)]
        # Find the minimum number of images left for each type
        min_leftover = min([l for l in leftovers if l > 0])
        for i in range(3):
            # Allocate the remaining images
            if leftovers[i] > 0 and remaining > 0:
                # Allocate the minimum of the remaining images, the minimum number of images left for each type, and the number of images left for each type
                to_allocate = min(min_leftover, remaining, leftovers[i])
                images_per_type[i] += to_allocate
                remaining -= to_allocate

    return images_per_type


def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

def count_top_samples(scores_dict, ratio=0.3):
    tagged_lists = []
    model_names = list(scores_dict.keys())
    for model_name in model_names:
        lst = scores_dict[model_name]
        tagged_lists.extend([(model_name, x) for x in lst])

    # 값을 기준으로 정렬 (두 번째 요소, 즉 x[1]을 기준으로)
    tagged_lists.sort(key=lambda x: x[1][1], reverse=True)

    # 상위 30% 요소 개수 계산
    top_count = int(len(tagged_lists) * ratio)

    # 상위 30% 요소 선택
    top_elements = tagged_lists[:top_count]

    result_dict = {}
    for model_name in model_names:
        samples_for_model = [x[1] for x in top_elements if x[0] == model_name]
        scores_list = [sample[1] for sample in samples_for_model]
        average_for_model = sum(scores_list) / len(scores_list)
        result_dict[model_name] = (len(scores_list), average_for_model)

    return result_dict

def get_topk_average(score_list, k):
    score_list.sort(reverse=True)
    if k == -1:
        pass
    else:
        score_list = score_list[:k]
    return sum(score_list) / len(score_list)


def normalize_and_clip_scores(scores):
    # Z-score normalization
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    normalized_scores = (scores - mean) / std

    # Clipping at the 0.025% and 99.975% quantiles
    lower_clip = np.percentile(normalized_scores, 0.025)
    upper_clip = np.percentile(normalized_scores, 99.975)
    clipped_scores = np.clip(normalized_scores, lower_clip, upper_clip)
    
    return clipped_scores

# Dataset class to load images
class ImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        return self.processor(images=image, return_tensors="pt")