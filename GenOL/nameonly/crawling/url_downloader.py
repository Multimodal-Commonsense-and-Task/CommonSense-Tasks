from img2dataset import download
import shutil
import os
import argparse
from classes import *
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--web_source", type=str)
    parser.add_argument("--start_class", type=int, default=0)
    parser.add_argument("--end_class", type=int)
    parser.add_argument("--min_images", type=int, default=1500)
    args = parser.parse_args()
    
    dataset_variable_dict = {'food101': food101_count}
    dataset = list(dataset_variable_dict[args.dataset_name].keys())
    
    if args.end_class is None:
        args.end_class = len(dataset) - 1
        print(f"End class not specified. Setting end class to {args.end_class}")
    
    classes = dataset[args.start_class:args.end_class + 1]
    print(f"Downloading classes starting from {args.start_class} to {args.end_class}")

    for cls in tqdm(classes):
        output_dir = os.path.join('datasets', f"{args.dataset_name}_{args.web_source}", cls)
        print(f"Downloading {cls} to {output_dir}")
        # Remove the directory if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        download(
            processes_count=8,
            thread_count=8,
            url_list=f"./urls/{args.dataset_name}_{args.web_source}/{cls}.txt",
            image_size=256,
            resize_mode="keep_ratio",
            output_folder=output_dir,
            output_format="files",
        )
    