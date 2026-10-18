import os
import shutil
from tqdm import tqdm

# flickr, google, bing
PATHS = [
    '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl_diversified',
    '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl_scaleup_ws',
    '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl_scaleup_sh',
    # '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/pacs_floyd'
]
OUTPUT_PATH = '/workspace/home/user/seongwon/crawling/crawler/datasets/PACS/PACS_sdxl_diversified_scaleup'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

image_extension = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPEG']

class_names = os.listdir(PATHS[0])
for class_name in tqdm(class_names):
    image_id = 0

    class_paths = [os.path.join(path, class_name) for path in PATHS]
    
    output_path = os.path.join(OUTPUT_PATH, class_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for path in class_paths:
        print(f"Processing {path}")
        for image in os.listdir(path):
            if image.endswith(tuple(image_extension)):
                image_name = str(image_id).zfill(6) + '.png'
                image_id += 1
                shutil.copy(os.path.join(path, image), os.path.join(output_path, image_name))


print('Done')