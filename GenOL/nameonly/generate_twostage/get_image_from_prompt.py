import os
import argparse
from get_image_onestage import model_selector
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True)
args = parser.parse_args()

root_dir = args.root_dir
model_name = "sdxl"
model = model_selector(model_name)

list_to_generate = ["A photo of a bird"] * 500

image_idx = 0
for prompt in tqdm(list_to_generate):
    try:
        image = model.generate_one_image(prompt)
    except Exception as e:
        print(f"Error occurred: {e}")
        continue
    
    image_path = os.path.join(root_dir, str(image_idx).zfill(6) + '.png')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    image.save(image_path, "JPEG")
    image_idx += 1

