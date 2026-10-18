import os
import sys
import json
import argparse
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, '../nameonly'))
sys.path.append(target_dir)
from classes import get_count_dict

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True)
args = parser.parse_args()

jsonl_data = []
# count_dict = get_count_dict(args.image_dir)
# classes = list(count_dict.keys())
classes = os.listdir(args.image_dir)
for cls in tqdm(classes):
        cls_path = os.path.join(args.image_dir, cls)
        for image_path in os.listdir(cls_path):
            jsonl_data.append({
                "file_name": os.path.join(cls, image_path),
                "caption": f"A photo of {cls}",
            })

output_path = os.path.join(args.image_dir, "metadata.jsonl")
with open(output_path, "w") as f:
    for data in jsonl_data:
        f.write(json.dumps(data) + "\n")