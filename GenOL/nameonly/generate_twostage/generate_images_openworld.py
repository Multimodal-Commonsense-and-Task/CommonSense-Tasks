import os
import json
import argparse
import time
import shutil
from get_image_onestage import model_selector
from get_image_onestage import adjust_list_length
from tqdm import tqdm
from utils import *
from pathlib import Path

debug = False

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str)
parser.add_argument('-r', '--root_dir', type=str)
parser.add_argument('-s', '--start_class', type=int)
parser.add_argument('-e', '--end_class', type=int)
parser.add_argument('-n', '--num_images', type=int, default=7)
args = parser.parse_args()

num_images = args.num_images

if debug:
    model = None
else:
    model = model_selector(args.model_name)

with open('../prompt_generation/prompts/openworld_diversified.json', 'r') as f:
    prompt_dict = json.load(f)

uid_list = list(prompt_dict.keys())
uids_to_generate = uid_list[args.start_class:args.end_class+1]

# Use queue to generate images for each class
queue_name = Path(args.root_dir).name
print(f"Set queue name as {queue_name}")
cls_initial_indices = list(range(args.start_class, args.end_class+1))
classes = [uid_list[i] for i in cls_initial_indices]
initialize_task_file(queue_name, args.start_class, args.end_class, classes)

while True:
    next_cls_idx = get_next_task(queue_name)
    if next_cls_idx is None:
        print(f"Task is done. Exiting...")
        break
    print(f"Class num {next_cls_idx} is selected. Start generating images for uid {uid_list[next_cls_idx]}")
    uid = uid_list[next_cls_idx]
    save_dir = os.path.join(args.root_dir, uid)
    # Remove existing directory
    if os.path.exists(save_dir):
        print(f"Removing existing directory: {save_dir}")
        shutil.rmtree(save_dir)
    pos_save_dir = os.path.join(save_dir, 'pos'); neg_save_dir = os.path.join(save_dir, 'neg')
    if not os.path.exists(pos_save_dir):
        os.makedirs(pos_save_dir)
    if not os.path.exists(neg_save_dir):
        os.makedirs(neg_save_dir)
    
    uid_prompts = prompt_dict[uid]
    
    # Change here to support the variable number of positive prompts
    positive_prompts = adjust_list_length(uid_prompts['positive_prompts'], num_images)
    negative_prompts = uid_prompts['negative_prompts']
    
    for i, pos_prompt in enumerate(positive_prompts):
        print(f"[POS] Generating image for uid:{uid}, prompt:{pos_prompt}")
        while True:
            try:
                image = model.generate_one_image(pos_prompt)
                break
            except:
                print(f"Error occurred. Retrying...")
                continue
        image_name = f"{sanitize_filename(pos_prompt)}_{str(i).zfill(6)}"
        image_path = os.path.join(pos_save_dir, image_name + '.png')
        image.save(image_path, "JPEG")
    
    for i, neg_prompt in enumerate(negative_prompts):
        print(f"[NEG] Generating image for uid:{uid}, prompt:{neg_prompt}")
        while True:
            try:
                image = model.generate_one_image(neg_prompt)
                break
            except:
                print(f"Error occurred. Retrying...")
                continue
        image_name = f"{sanitize_filename(neg_prompt)}_{str(i).zfill(6)}"
        image_path = os.path.join(neg_save_dir, image_name + '.png')
        image.save(image_path, "JPEG")

    mark_task_done(queue_name, next_cls_idx)