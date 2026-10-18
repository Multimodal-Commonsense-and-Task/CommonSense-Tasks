# 0516
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

prompt_path = './prompts/static_totalprompts_with_cot_100_with_cls.json'
output_path = './prompts/static_totalprompts_with_cot_100_with_cls_filtered_50.json'

with open(prompt_path, 'r') as f:
    total_prompt_dict = json.load(f)

model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Concatenetate the static prompt and diversified prompt
classes = list(PACS_count.keys())
for cls in classes:
    prompt_list = []
    for metaprompt_dict in total_prompt_dict[cls]['metaprompts']:
        for prompt_dict in metaprompt_dict['prompts']:
            prompt_info = (metaprompt_dict['index'], prompt_dict['index'], prompt_dict['content'])
            prompt_list.append(prompt_info) # (metaprompt_idx, prompt_idx, prompt)
    
    # Clip filter 100 prompts to select top 50
    text_list = [prompt_info[2] for prompt_info in prompt_list]
    reference_feature = get_text_features(model, f"A realistic photo of {cls}")
    text_features = get_text_features(model, text_list)
    similarity = reference_feature @ text_features.T
    rank = similarity.argsort(descending=True).cpu().numpy()[0]
    sorted_prompts = [prompt_list[i] for i in rank]
    selected_prompts = sorted_prompts[50:]

    # Update the result dict
    # Remove the selected (bottom-50) prompts
    for prompt_info in selected_prompts:
        # Find the prompt
        temp = total_prompt_dict[cls]['metaprompts'][prompt_info[0]]['prompts']
        for prompt_dict in temp:
            if prompt_dict['index'] == prompt_info[1]:
                temp.remove(prompt_dict)
        
with open(output_path, 'w') as f:
    json.dump(total_prompt_dict, f)
