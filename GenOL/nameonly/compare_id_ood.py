import os
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, ViTModel
from sklearn.metrics.pairwise import cosine_similarity

DINO_Transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform = transforms.Compose([
    transforms.ToTensor()
])

def calculate_features_CLIP(image_paths, model, processor, device, normalize=False):
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

def calculate_features_DINO(image_paths, model, device, normalize=False):
    # Preprocess the images
    image_features = []
    for i, image_path in enumerate(tqdm(image_paths)):
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            image_embedding = model(DINO_Transform(image).unsqueeze(0).to(device)).last_hidden_state[0, 0]
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True) # Normalize
            image_features.append(image_embedding.detach().cpu().numpy())

    if normalize:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Send the features to the cpu
    return image_features

def get_dino_similarities(self, dino_model, images_src, images_tgt):
    src_embeddings = []; tgt_embeddings = []
    with torch.no_grad():
        for image_dict in tqdm(images_src):
            image = image_dict["image"]
            image_embedding = dino_model(DINO_Transform(image).unsqueeze(0).to(self.device)).last_hidden_state[0, 0]
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            src_embeddings.append(image_embedding.detach().cpu().numpy())
        
        for image_dict in tqdm(images_tgt):
            image = image_dict["image"]
            image_embedding = dino_model(DINO_Transform(image).unsqueeze(0).to(self.device)).last_hidden_state[0, 0]
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            tgt_embeddings.append(image_embedding.detach().cpu().numpy())

    # Reshaping and calculating cosine similarities
    src_embeddings = np.array(src_embeddings)
    src_embeddings = src_embeddings.reshape(len(src_embeddings), -1)
    tgt_embeddings = np.array(tgt_embeddings)
    tgt_embeddings = tgt_embeddings.reshape(len(tgt_embeddings), -1)
    src_tgt_cos_similarities = cosine_similarity(src_embeddings, tgt_embeddings)
    dino_average_similarities = np.mean(src_tgt_cos_similarities, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain1', type=str)
    parser.add_argument('--domain2', type=str)
    parser.add_argument('--model', type=str, default='dino')

    args = parser.parse_args()

    domain1_path = args.domain1; domain2_path = args.domain2
    classes = os.listdir(domain1_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load CLIP model
    if args.model == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    elif args.model == 'dino':
        model = ViTModel.from_pretrained("facebook/dino-vits16").to(device)
    
    cls_similariy_dict = {}
    for cls in tqdm(classes):
        if cls.endswith('.json'):
            continue
        domain1_cls_path = os.path.join(domain1_path, cls)
        domain2_cls_path = os.path.join(domain2_path, cls)
        domain1_images = [os.path.join(domain1_cls_path, image_path) for image_path in os.listdir(domain1_cls_path) if not image_path.endswith('.json')]
        domain2_images = [os.path.join(domain2_cls_path, image_path) for image_path in os.listdir(domain2_cls_path) if not image_path.endswith('.json')]

        if args.model == 'clip':
            domain1_features = calculate_features_CLIP(domain1_images, model, processor, device)
            domain2_features = calculate_features_CLIP(domain2_images, model, processor, device)

            # Calculate cosine similarity of each class
            similarity_matrix = cosine_similarity(domain1_features, domain2_features)
            average_similarity = np.mean(similarity_matrix)
        
            cls_similariy_dict[cls] = average_similarity
        
        elif args.model == 'dino':
            domain1_features = calculate_features_DINO(domain1_images, model, device)
            domain2_features = calculate_features_DINO(domain2_images, model, device)

            # Reshaping and calculating cosine similarities
            domain1_embeddings = np.array(domain1_features)
            domain1_embeddings = domain1_embeddings.reshape(len(domain1_embeddings), -1)
            domain2_embeddings = np.array(domain2_features)
            domain2_embeddings = domain2_embeddings.reshape(len(domain2_embeddings), -1)
            domain1_domain2_cos_similarities = cosine_similarity(domain1_embeddings, domain2_embeddings)
            dino_average_similarities = np.mean(domain1_domain2_cos_similarities)

            cls_similariy_dict[cls] = dino_average_similarities
    
    print(f"Average cosine similarity of two domains: {np.mean(list(cls_similariy_dict.values()))}")