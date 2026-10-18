# Nested prompt structure -> list of prompts
import json
from classes import *

prompt_path = './prompts/synthclip_cifar10.json'
output_path = './quali_prompts/synthclip_cifar10_list.json'
count_dict = cifar10_count

with open(prompt_path, 'r') as f:
    data_dict = json.load(f)

result_dict = {}
if 'metaprompts' in data_dict:
    # Static prompts
    for cls in count_dict.keys():
        result_dict[cls] = []
        for metaprompt_dict in data_dict['metaprompts']:
            for prompt_dict in metaprompt_dict['prompts']:
                result_dict[cls].append(prompt_dict['content'].replace('[concept]', cls.replace('_', ' ')))
else:
    # Dynamic prompts
    for cls in data_dict:
        result_dict[cls] = []
        for metaprompt_dict in data_dict[cls]['metaprompts']:
            for prompt_dict in metaprompt_dict['prompts']:
                result_dict[cls].append(prompt_dict['content'])

with open(output_path, 'w') as f:
    json.dump(result_dict, f, indent=4)
