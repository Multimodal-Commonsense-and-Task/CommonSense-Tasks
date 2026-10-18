from img2dataset import download
import shutil
import os
import argparse
from classes import *
from tqdm import tqdm

url_path = './urls/bongard_hoi'
url_list = os.listdir(url_path)

for url_file in tqdm(url_list):
    id = url_file.split('.')[0]
    output_dir = os.path.join('datasets', f"bongard_hoi", id)
    print(f"Downloading {id} to {output_dir}")
    # Remove the directory if it exists
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # elif not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    url_file = os.path.join(url_path, url_file)
    
    if __name__ == "__main__":
        # from img2dataset import download
        download(
            thread_count=64,
            url_list=url_file,
            # image_size=256,
            resize_mode="keep_ratio",
            output_folder=output_dir,
            output_format="files",
        )