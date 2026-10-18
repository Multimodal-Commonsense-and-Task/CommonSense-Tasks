import os
import shutil
import random
from tqdm import tqdm
from classes import *
from utils import distribute_equally

count_dict = cifar10_count

# flickr, google, bing
flickr_PATH = '/workspace/home/user/seongwon/crawling/crawler/datasets/cifar10/cifar10_flickr'
google_PATH = '/workspace/home/user/seongwon/crawling/crawler/datasets/cifar10/cifar10_google'
bing_PATH = '/workspace/home/user/seongwon/crawling/crawler/datasets/cifar10/cifar10_bing'

OUTPUT_PATH = '/workspace/home/user/seongwon/crawling/crawler/datasets/cifar10/cifar10_web_ratio_preserved_1_1'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

image_extension = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPEG']

class_names = count_dict.keys()
flickr_images_dict = {cls: os.listdir(os.path.join(flickr_PATH, cls)) for cls in class_names}
google_images_dict = {cls: os.listdir(os.path.join(google_PATH, cls)) for cls in class_names}
bing_images_dict = {cls: os.listdir(os.path.join(bing_PATH, cls)) for cls in class_names}

# Randomly shuffle the images
for cls in class_names:
    random.shuffle(flickr_images_dict[cls])
    random.shuffle(google_images_dict[cls])
    random.shuffle(bing_images_dict[cls])

for cls in tqdm(class_names):
    target_num = int(count_dict[cls] * 1.0)
    if len(flickr_images_dict[cls]) + len(google_images_dict[cls]) + len(bing_images_dict[cls]) < target_num:
        raise ValueError(f"Class {cls} does not have enough images")

    # First find the number of images to be taken from each source
    num_google_crawled = len(google_images_dict[cls])
    num_bing_crawled = len(bing_images_dict[cls])
    num_flickr_crawled = len(flickr_images_dict[cls])

    google_count, bing_count, flickr_count = distribute_equally(num_google_crawled, num_bing_crawled, num_flickr_crawled, target_num)
    print(f"Class {cls}: {google_count} from Google, {bing_count} from Bing, {flickr_count} from Flickr")

    # Create the output directory
    output_path = os.path.join(OUTPUT_PATH, cls)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Move the images to the output directory
    index = 0
    for i in range(google_count):
        image_name = 'google_' + str(index).zfill(6) + '.png'
        index += 1
        shutil.copy(os.path.join(google_PATH, cls, google_images_dict[cls][i]), os.path.join(OUTPUT_PATH, cls, image_name))
    for i in range(bing_count):
        image_name = 'bing_' + str(index).zfill(6) + '.png'
        index += 1
        shutil.copy(os.path.join(bing_PATH, cls, bing_images_dict[cls][i]), os.path.join(OUTPUT_PATH, cls, image_name))
    for i in range(flickr_count):
        image_name = 'flickr_' + str(index).zfill(6) + '.png'
        index += 1
        shutil.copy(os.path.join(flickr_PATH, cls, flickr_images_dict[cls][i]), os.path.join(OUTPUT_PATH, cls, image_name))
print('Done')