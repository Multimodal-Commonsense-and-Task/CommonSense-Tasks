# Assign class name to static prompt
import json
from copy import deepcopy
from classes import *

# static prompt는 class name이 없는데, image generation code는 class - prompt dict로
# 되어 있기 때문에 이를 수정하는 부분.

classes = list(cifar10_count.keys())
static_path = './prompts/gpt4_hierarchy_cot_50_2.json'
output_path = './quali_prompts/gpt4_hierarchy_cot_50_2_cifar10.json'

with open(static_path, 'r') as f:
    static_dict = json.load(f)

static_to_dynamic_dict = {}
for cls in classes:
    static_to_dynamic_dict[cls] = {'metaprompts': deepcopy(static_dict['metaprompts'])}

# Conver to class name
for k, v in static_to_dynamic_dict.items():
    metaprompt_list = v['metaprompts']
    for metaprompt_dict in metaprompt_list:
        if '[concept]' in metaprompt_dict['metaprompt']:
            metaprompt_dict['metaprompt'] = metaprompt_dict['metaprompt'].replace("[concept]", k)

        for prompt_dict in metaprompt_dict['prompts']:
            assert "[concept]" in prompt_dict['content'], f"No [concept] in {prompt_dict}"
            prompt_dict['content'] = prompt_dict['content'].replace("[concept]", k.replace('_', ' '))

with open(output_path, 'w') as f:
    json.dump(static_to_dynamic_dict, f)