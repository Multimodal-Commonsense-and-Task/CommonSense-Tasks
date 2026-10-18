import os
import json
import argparse
import shutil
from get_image_onestage import model_selector, adjust_list_length
from tqdm import tqdm

debug = False

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str)
parser.add_argument('-r', '--root_dir', type=str)
parser.add_argument('-s', '--start_class', type=int)
parser.add_argument('-e', '--end_class', type=int)
parser.add_argument('-p', '--prompt_dir')
args = parser.parse_args()

if debug:
    model = None
else:
    model = model_selector(args.model_name)

# Load prompt json
with open(args.prompt_dir, 'r') as f:
    dataset_list = json.load(f)

print(f"Class indices to generate: {args.start_class} ~ {args.end_class}")
list_to_generate = dataset_list[args.start_class:args.end_class+1]
for dataset_dict in tqdm(list_to_generate):
    object_name = dataset_dict['object_name']
    action_name = dataset_dict['action_name']
    prompts = dataset_dict['prompts']
    num_images = dataset_dict['count']
    
    save_dir = os.path.join(args.root_dir, object_name, action_name)
    # Remove existing directory
    if os.path.exists(save_dir):
        print(f"Removing existing directory: {save_dir}")
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Start generating {num_images} images.")
    
    prompts = adjust_list_length(prompts, num_images)
    for i in tqdm(range(num_images)):
        print(f"Generating image for {object_name} - {action_name}, prompt: {prompts[i]}")
        while True:
            try:
                image = model.generate_one_image(prompts[i])
                break
            except:
                print(f"Error occurred. Retrying...")
                continue
        image_path = os.path.join(save_dir, str(i).zfill(6) + '.png')
        image.save(image_path, "JPEG")
    