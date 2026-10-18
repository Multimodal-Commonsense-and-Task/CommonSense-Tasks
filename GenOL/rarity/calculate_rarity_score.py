import os
import random
from rarity_score import *
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import models, transforms

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--real_path', required=True)
parser.add_argument('--generated_path', required=True)
parser.add_argument('--sample_size', type=int, default=100)
args = parser.parse_args()

result = calculate_rarity_score(args.real_path, args.generated_path, sample_size=args.sample_size)
