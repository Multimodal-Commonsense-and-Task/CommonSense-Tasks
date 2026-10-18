import os
import json
from tqdm import tqdm

prompt_path = "../prompt_generation/prompts/generated_LE_ver1.json"
root_dir = "../raw_datasets/iclr_generated/hoi/generated_LE_ver1_floyd"

with open("../prompt_generation/prompts/generated_LE_ver1.json", "r") as f:
    prompts = json.load(f)

uid_list = [str(data['id']) for data in prompts]
cls_dir_count_dict = {}
cls_not_generated = []

for i, uid in enumerate(tqdm(uid_list)):
    pos_dir = os.path.join(root_dir, uid, 'pos')
    neg_dir = os.path.join(root_dir, uid, 'neg')
    if os.path.exists(pos_dir):
        pos_count = len(os.listdir(pos_dir))
    else:
        pos_count = 0
    if os.path.exists(neg_dir):
        neg_count = len(os.listdir(neg_dir))
    else:
        neg_count = 0
    
    cls_dir_count_dict[i] = (uid, pos_count, neg_count)

# Visualize indices of classes that are not generated
for i, (uid, pos_count, neg_count) in cls_dir_count_dict.items():
    if pos_count != 7 or neg_count != 7:
        print(f"{i}: {uid} - pos: {pos_count}, neg: {neg_count}")
        cls_not_generated.append(i)

print(cls_not_generated)
print(f"Total number of classes not generated: {len(cls_not_generated)}")
print(f"Number of classes generated: {len(uid_list) - len(cls_not_generated)}")