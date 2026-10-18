import os
import json

prompt_path = '../prompt_generation/prompts/openworld_diversified.json'
root_dir = './generated_datasets/openworld_diversified_auraflow'

with open('../prompt_generation/prompts/openworld_diversified.json', 'r') as f:
    prompt_dict = json.load(f)

uid_list = list(prompt_dict.keys())

cls_dir_count_dict = {}
cls_not_generated = []

for i, uid in enumerate(uid_list):
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

print(f"Total number of classes not generated: {len(cls_not_generated)}")
print(cls_not_generated)