import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, multi_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        if multi_train:
            real_root_dir = root_dir
            for domain_name in os.listdir(real_root_dir):
                self.root_dir = os.path.join(real_root_dir, domain_name)
                self._load_data()
                
        else:
            self._load_data()
        
    def _load_data(self):
        for label, class_name in enumerate(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
