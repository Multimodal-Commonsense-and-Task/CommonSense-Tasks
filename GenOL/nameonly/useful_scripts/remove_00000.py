import os
import shutil
from tqdm import tqdm
from classes import *

count_dict = cct_count
path = '/home/user/seongwon/nameonly/raw_datasets/web/cct/cct_bing'
target_path = '/home/user/seongwon/nameonly/raw_datasets/web/cct/cct_bing_removed_00000'

image_extension = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPEG']
classes = count_dict.keys()
for cls in tqdm(classes):
    cls_path = os.path.join(path, cls, '00000')
    target_cls_path = os.path.join(target_path, cls)
    if not os.path.exists(target_cls_path):
        os.makedirs(target_cls_path)
    for image in os.listdir(cls_path):
        if image.endswith(tuple(image_extension)):
            shutil.copy(os.path.join(cls_path, image), os.path.join(target_cls_path, image))
