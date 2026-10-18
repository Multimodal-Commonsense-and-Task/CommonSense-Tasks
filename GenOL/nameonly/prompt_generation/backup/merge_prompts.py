# Merge base prompt and classwise prompt (0514 ~ 0515)
import json
import torch
import clip
from classes import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def get_text_features(model, strings: list, norm=True):
    tokens = clip.tokenize(strings).to(device)
    text_features = model.encode_text(tokens)
    
    if norm:
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

static_path = './prompts/static_totalprompts.json'
dynamic_path = './prompts/dynamic_totalprompts_PACS.json'

with open(static_path, 'r') as f:
    static_dict = json.load(f)

with open(dynamic_path, 'r') as f:
    dynamic_dict = json.load(f)

model, clip_preprocess = clip.load("ViT-B/32", device=device)
# 

# # Get diversified prompts of each dict
# for k, v in static_dict.items():
#     print(k)
#     for diversified_dict in v['prompts']:
#         if '[concept]' not in diversified_dict['content']:
#             breakpoint()

# Concatenetate the static prompt and diversified prompt
classes = list(PACS_count.keys())
class_prompt_dict = {}
for cls in classes:
    prompt_list = []
    for k, v in static_dict.items():
        for diversified_dict in v['prompts']:
            if '[concept]' not in diversified_dict['content']:
                breakpoint()
            prompt = {
                'metaprompt_idx': v['index'],
                'diversified_prompt_idx': diversified_dict['index'],
                'content': diversified_dict['content'].replace('[concept]', cls)
            }
            prompt_list.append(prompt)
    
    for metaprompt_dict in dynamic_dict[cls]:
        metaprompt = metaprompt_dict['metaprompt']
        metaprompt_index = metaprompt_dict['index']
        for prompt_dict in metaprompt_dict['prompts']:
            prompt = {
                'metaprompt_idx': metaprompt_index,
                'diversified_prompt_idx': prompt_dict['index'],
                'content': prompt_dict['content']
            }
            prompt_list.append(prompt)

    # Clip filter 100 prompts to select top 50
    text_list = [prompt_dict['content'] for prompt_dict in prompt_list]
    reference_feature = get_text_features(model, f"A realistic {cls}")
    text_features = get_text_features(model, text_list)
    similarity = reference_feature @ text_features.T
    rank = similarity.argsort(descending=True).cpu().numpy()[0]
    sorted_prompts = [prompt_list[i] for i in rank]
    selected_prompts = sorted_prompts[:50]

    class_prompt_dict[cls] = selected_prompts

with open('./prompts/clip_filtered_prompts.json', 'w') as f:
    json.dump(class_prompt_dict, f)
