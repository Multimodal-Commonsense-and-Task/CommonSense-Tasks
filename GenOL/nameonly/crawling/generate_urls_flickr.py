import subprocess
import time
import os
import logging
import json
from url_generator.flickr_generator import FlickrURLGenerator
from classes import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEYS=(
    "da7938344465f98fa66ce93d3797151b",# 0
    "4737397d3f91a4c7a73b1a05aab2f5a1",# 1
    "002a4457c6886047f3384f446a3753ef",# 2
    "3a21a9a2b34ccd30ea902767ebb9b804",# 3
    "3ce653afa0cea5cf77c2f829ffbe224b",# 4
    "05265e4d91eb28d722dcc29bbf464e0e",# 5
    "36d5370627afb46a46f634bf6b6b46fd",# 6
    "f7b5ba95d3c585b8a4e4dea513f43da5",# 7
    "3588e8771d5d11e30a6f08bf7c293436",# 8
    "d22c514a40ff4a2b5732d0199c9c91b2",# 9
    "ba926e463a9d57fc36e5b45a470aff64",# 10
    "378be5b24af3c62db644553906bd6190",# 11
    "67c66c958eb6e034f37cf4dba30a62b8",# 12
    "fb36057a4a802232883d8888d248d86f",# 13
    "cd6d8b6a6cb7890175bdeb7cb1e38924",# 14
    "52653a2dba376c534eaddd73f71f9918",# 15
    "2332a55df4caa40dffbc43811f0b0ad5",# 16
    "4d23fddceb90eb28fe177a25e32eb22b",# 17
    "075e07f82d0b4051b382448ea96a27a4",# 18
    "6e2f4ef23fa563e659eb109cb0306125",# 19
    "c23a36a8b8a91b373f6358747cd17d65",# 20
    "2626778b7c7c12de00aaba7b7d3e1d77",# 21
    "41de4d7068eef7473921f778779256d5",# 22
)

save_dir = "urls/Bongard_flickr"

flickr_generator = FlickrURLGenerator(
    api_key=API_KEYS[5],
    save_dir=save_dir,
    error_dir="error",
    max_page=5,
)

# for cls in tqdm(PACS_count):
#     logger.info(f"Generating URL for {cls}")
#     result = flickr_generator.generate_url(keyword=cls, total_images=7000, images_per_date_range=4000)


# Process jsonl file
data_list = []
with open('../generate_twostage/train.jsonl', 'r') as f:
    for line in f:
        json_object = json.loads(line)
        data_list.append(json_object)

for data in tqdm(data_list):
    caption = data['caption']
    uid = data['uid']
    logger.info(f"Generating URL for uid {uid} - {caption}")
    result = flickr_generator.generate_url(keyword=caption, total_images=100, images_per_date_range=100, filename=uid)
