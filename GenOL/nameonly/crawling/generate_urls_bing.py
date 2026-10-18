import os
import logging
from url_generator.bing_generator import BingURLGenerator
from classes import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

save_dir = './urls/domainnet_urls_bing'

generator = BingURLGenerator(mode='headless', use_color=False, use_size=False, scroll_patience=20)

# # Check the save directory to remove already generated URLs
# class_txt_files = os.listdir(save_dir)
# class_txt_files = [f.split('.')[0] for f in class_txt_files]
# domainnet = [cls for cls in domainnet if cls not in class_txt_files]

# For classification
for cls in tqdm(DomainNet_count):
    logger.info(f"Generating URL for {cls}")
    url_list = generator.generate_url(query=cls, total_images=10, image_type='None')
    with open(os.path.join(save_dir, f'{cls}.txt'), 'w') as f:
        for url in url_list:
            f.write(url + '\n')
