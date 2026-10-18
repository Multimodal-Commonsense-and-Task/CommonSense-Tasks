import os
import shutil
import random

base_path = './datasets/openworld_processed'
target_path = './datasets/openworld_web'
NUM_IMAGES = 7

url_list = os.listdir(base_path)
for url in url_list:
    url_source_path = os.path.join(base_path, url)
    url_source_path_pos = os.path.join(url_source_path, 'pos')
    url_source_path_neg = os.path.join(url_source_path, 'neg')
    
    pos_images = [os.path.join(url_source_path_pos, image) for image in os.listdir(url_source_path_pos)]
    neg_images = [os.path.join(url_source_path_neg, image) for image in os.listdir(url_source_path_neg)]
    pos_images = random.sample(pos_images, NUM_IMAGES)
    neg_images = random.sample(neg_images, NUM_IMAGES)
    
    url_target_path_pos = os.path.join(target_path, url, 'pos')
    url_target_path_neg = os.path.join(target_path, url, 'neg')
    os.makedirs(url_target_path_pos, exist_ok=True); os.makedirs(url_target_path_neg, exist_ok=True)

    for image in pos_images:
        image_name = os.path.basename(image)
        target_image_path_pos = os.path.join(url_target_path_pos, image_name)
        shutil.copy(image, target_image_path_pos)
    for image in neg_images:
        image_name = os.path.basename(image)
        target_image_path_neg = os.path.join(url_target_path_neg, image_name)
        shutil.copy(image, target_image_path_neg)