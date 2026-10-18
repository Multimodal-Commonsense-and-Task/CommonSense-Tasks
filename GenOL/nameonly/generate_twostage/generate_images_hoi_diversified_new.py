import os
import json
import argparse
import signal
import shutil
import time
import re
from get_image_queue import model_selector, adjust_list_length
from tqdm import tqdm
from utils import *
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str)
parser.add_argument('-r', '--root_dir', type=str)
parser.add_argument('-s', '--start_class', type=int)
parser.add_argument('-e', '--end_class', type=int)
parser.add_argument('-p', '--prompt_dir')
args = parser.parse_args()

debug = False

if debug:
    model = None
else:
    model = model_selector(args.model_name)

# Load prompt json
with open(args.prompt_dir, 'r') as f:
    dataset_list = json.load(f)

print(f"ID indices to generate: {args.start_class} ~ {args.end_class}")
list_to_generate = dataset_list[args.start_class:args.end_class+1]

# Use queue to generate images for each class
queue_name = Path(args.root_dir).name
print(f"Set queue name as {queue_name}")
cls_initial_indices = list(range(args.start_class, args.end_class+1))
classes = [dataset_list[i]['id'] for i in cls_initial_indices]
initialize_task_file(queue_name, args.start_class, args.end_class, classes)

def signal_handler(sig, frame, queue_name, current_cls_id):
    # Read and lock the task file
    task_file = f"{queue_name}_task.txt"
    f = open(task_file, 'r+')
    fcntl.flock(f, fcntl.LOCK_EX)
    content = f.readlines()
    
    # Mark the current task as pending_sigkill
    for i in range(len(content)):
        cls_idx, cls_id, status = content[i].strip().split()
        cls_idx = int(cls_idx)
        if cls_idx == current_cls_id:
            content[i] = f"{cls_idx} {cls_id} pending_preempted_or_killed\n"
            break
    
    # Write and unlock the task file
    f.seek(0)
    f.writelines(content)
    f.truncate()
    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()
    
    print(f"Received signal {sig}. Exiting...")
    os._exit(0)

# Set signal handler
signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, queue_name, next_cls_idx))
signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, queue_name, next_cls_idx))
signal.signal(signal.SIGUSR1, lambda sig, frame: signal_handler(sig, frame, queue_name, next_cls_idx))

while True:
    next_cls_idx = get_next_task(queue_name) # This is from 0 ~ len(dataset_list)
    if next_cls_idx is None:
        print(f"Task is None. Exiting...")
        break
    print(f"Class num {next_cls_idx} is selected. Start generating images for id {dataset_list[next_cls_idx]['id']}")
    dataset_dict = dataset_list[next_cls_idx]
    id = dataset_dict['id']
    positive_prompts = dataset_dict['positive_prompts']
    negative_prompts = dataset_dict['negative_prompts']
    
    # Generate positive images
    regenrate_pos = False
    pos_save_dir = os.path.join(args.root_dir, str(id), 'pos')
    if os.path.exists(pos_save_dir):
        if len(os.listdir(pos_save_dir)) == len(positive_prompts):
            print(f"Skipping existing directory: {pos_save_dir}")
        else:
            print(f"Removing existing directory as the number of images is different: {pos_save_dir}")
            shutil.rmtree(pos_save_dir)
            os.makedirs(pos_save_dir, exist_ok=True)
            regenrate_pos = True
    else:
        os.makedirs(pos_save_dir, exist_ok=True)
        regenrate_pos = True
    if regenrate_pos:
        for i, positive_prompt in enumerate(positive_prompts):
            print(f"Generating positive image for {id}, prompt: {positive_prompt}")
            while True:
                try:
                    image = model.generate_one_image(positive_prompt)
                    break
                except:
                    print(f"Error occurred. Retrying...")
                    continue
            image_path = os.path.join(pos_save_dir, str(i).zfill(6) + '.png')
            image.save(image_path, "JPEG")
    
    # Generate negative images
    neg_save_dir = os.path.join(args.root_dir, str(id), 'neg')
    regenrate_neg = False
    if os.path.exists(neg_save_dir):
        if len(os.listdir(neg_save_dir)) == len(negative_prompts):
            print(f"Skipping existing directory: {neg_save_dir}")
        else:
            print(f"Removing existing directory as the number of images is different: {neg_save_dir}")
            shutil.rmtree(neg_save_dir)
            os.makedirs(neg_save_dir, exist_ok=True)
            regenrate_neg = True
    else:
        os.makedirs(neg_save_dir, exist_ok=True)
        regenrate_neg = True
    if regenrate_neg:
        for i, negative_prompt in enumerate(negative_prompts):
            print(f"Generating negative image for {id}, prompt: {negative_prompt}")
            while True:
                try:
                    image = model.generate_one_image(negative_prompt)
                    break
                except:
                    print(f"Error occurred. Retrying...")
                    continue
            image_path = os.path.join(neg_save_dir, str(i).zfill(6) + '.png')
            image.save(image_path, "JPEG")
    # Mark the task as completed
    mark_task_done(queue_name, next_cls_idx)
