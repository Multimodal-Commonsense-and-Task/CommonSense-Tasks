# Get validation set using hash function

from PIL import Image
import argparse
import hashlib
import os
import hashlib
import shutil
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path')
    parser.add_argument('-t', '--target_path')
    parser.add_argument('-r', '--result_save_path')
    args = parser.parse_args()

    classes = os.listdir(args.source_path)
    classes = [cls for cls in classes if not cls.endswith(".json")]

    
    for cls in tqdm(classes):
        source_path = os.path.join(args.source_path, cls)
        target_path = os.path.join(args.target_path, cls)
        cls_save_path = os.path.join(args.result_save_path, cls)
        if not os.path.exists(cls_save_path):
            os.makedirs(cls_save_path, exist_ok=True)
        
        source_images = os.listdir(source_path); target_images = os.listdir(target_path)
        source_images = [os.path.join(source_path, image) for image in source_images]
        target_images = [os.path.join(target_path, image) for image in target_images]

        # Get all target hashes
        target_hashes = []
        for image in target_images:
            with Image.open(image) as img:
                # Convert image to grayscale (optional for further efficiency)
                img = img.convert('L')
                # Get image data as a byte array
                image_data = img.tobytes()
                hash_value = hashlib.sha256(image_data).hexdigest()
                target_hashes.append(hash_value)
        
        source_not_in_targets = []
        for image in source_images:
            with Image.open(image) as img:
                # Convert image to grayscale (optional for further efficiency)
                img = img.convert('L')
                image_data = img.tobytes()
                hash_value = hashlib.sha256(image_data).hexdigest()
                if hash_value not in target_hashes:
                    source_not_in_targets.append(image)

        # Save images that are not in target images
        for i, image in enumerate(source_not_in_targets):
            save_path = os.path.join(cls_save_path, f"{str(i).zfill(5)}.png")
            Image.open(image).save(save_path)