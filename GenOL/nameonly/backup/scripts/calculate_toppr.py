# Call packages
import matplotlib.pyplot as plot
import numpy as np
import torch
import os
import natsort

# Call mode drop example case
from top_pr import mode_drop

# Call metrics
from top_pr import compute_top_pr as TopPR
# For comparison to PRDC, use this. 'pip install prdc'
from prdc import compute_prdc
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image


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

def calculate_features_CLIP(image_paths, model, processor, device):
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

def calculate_features_vgg(image_paths, device):
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
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Send the features to the cpu
    image_features = image_features.cpu().detach().numpy()
    return image_features

def calculate_toppr_score(real_path, generated_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    images_real = natsort.natsorted(os.listdir(real_path))
    images_real = [os.path.join(real_path, image) for image in images_real]
    images_generated = natsort.natsorted(os.listdir(generated_path))
    images_generated = [os.path.join(generated_path, image) for image in images_generated]
    real_features = calculate_features_CLIP(images_real, model, processor, device)
    generated_features = calculate_features_CLIP(images_generated, model, processor, device)
    # real_features = calculate_features_vgg(images_real, device)
    # generated_features = calculate_features_vgg(images_generated, device)
    
    # Calculate TopPR
    Top_PR = TopPR(real_features=real_features, fake_features=generated_features, alpha = 0.1, kernel = "cosine", random_proj = True, f1_score = True)
    
    return Top_PR

# real_path = './datasets/ablations/cifar10_MA/horse'
real_path = './datasets/PACS/PACS_MA/photo/dog'
# generated_path = './datasets/ablations/cifar10_full/horse'
generated_path = './raw_datasets/generated/PACS/PACS_sdxl_diversified/dog'
score = calculate_toppr_score(real_path, generated_path)
breakpoint()

# Evaluation step
start = 0
for Ratio in [0, 1, 2, 3, 4, 5, 6]:

    # Define real and fake dataset
    REAL = mode_drop.gaussian_mode_drop(method = 'sequential', ratio = 0)
    FAKE = mode_drop.gaussian_mode_drop(method = 'sequential', ratio = Ratio)
    breakpoint()
    # Evaluation with TopPR
    Top_PR = TopPR(real_features=REAL, fake_features=FAKE, alpha = 0.1, kernel = "cosine", random_proj = True, f1_score = True)
        
    # Evaluation with P&R and D&C
    PR = compute_prdc(REAL, FAKE, 3)
    DC = compute_prdc(REAL, FAKE, 5)
        
    if (start == 0):
        pr = [PR.get('precision'), PR.get('recall')]
        dc = [DC.get('density'), DC.get('coverage')]
        Top_pr = [Top_PR.get('fidelity'), Top_PR.get('diversity'), Top_PR.get('Top_F1')]
        start = 1
            
    else:
        pr = np.vstack((pr, [PR.get('precision'), PR.get('recall')]))
        dc = np.vstack((dc, [DC.get('density'), DC.get('coverage')]))
        Top_pr = np.vstack((Top_pr, [Top_PR.get('fidelity'), Top_PR.get('diversity'), Top_PR.get('Top_F1')]))

# Visualization of Result
x = [0, 0.17, 0.34, 0.50, 0.67, 0.85, 1]
fig = plot.figure(figsize = (12,3))
for i in range(1,3):
    axes = fig.add_subplot(1,2,i)
    
    # Fidelity
    if (i == 1):
        axes.set_title("Fidelity",fontsize = 15)
        plot.ylim([0.5, 1.5])
        plot.plot(x, Top_pr[:,0], color = [255/255, 110/255, 97/255], linestyle = '-', linewidth = 3, marker = 'o', label = "TopP")
        plot.plot(x, pr[:,0], color = [77/255, 110/255, 111/255], linestyle = ':', linewidth = 3, marker = 'o', label = "precision (k=3)")
        plot.plot(x, dc[:,0], color = [15/255, 76/255, 130/255], linestyle = '-.', linewidth = 3, marker = 'o', label = "density (k=5)")
        plot.plot(x, np.linspace(1.0, 1.0, 11), color = 'black', linestyle = ':', linewidth = 2)
        plot.legend(fontsize = 9)
    
    # Diversity
    elif (i == 2):
        axes.set_title("Diversity",fontsize = 15)
        plot.plot(x, Top_pr[:,1], color = [255/255, 110/255, 97/255], linestyle = '-', linewidth = 3, marker = 'o', label = "TopR")
        plot.plot(x, pr[:,1], color = [77/255, 110/255, 111/255], linestyle = ':', linewidth = 3, marker = 'o', label = "recall (k=3)")
        plot.plot(x, dc[:,1], color = [15/255, 76/255, 130/255], linestyle = '-.', linewidth = 3, marker = 'o', label = "coverage (k=5)")
        plot.plot(x, np.linspace(1.0, 0.14, 11), color = 'black', linestyle = ':', linewidth = 2)
        plot.legend(fontsize = 9)

fig.savefig('test.png')