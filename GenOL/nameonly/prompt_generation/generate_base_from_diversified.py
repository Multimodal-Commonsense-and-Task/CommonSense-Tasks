# 민혁님께서 json 달라고 하신 것 (txt file을 json으로 바꿈)
import json
from classes import *

input_json_path = './prompts/gpt_static_totalprompts_with_cot_50_with_cls_NICO.json'
output_json_path = './prompts/gpt_static_totalprompts_with_cot_50_with_cls_NICO_base.json'

with open(input_json_path, 'r') as f:
    input_dict = json.load(f)

json_dict = {}
for cls_name, metaprompts_dict in input_dict.items():
    metaprompts_list = metaprompts_dict['metaprompts']
    json_dict[cls_name] = {
        'metaprompts': []
    }
    for metaprompt_dict in metaprompts_list:
        json_dict[cls_name]['metaprompts'].append({
            'index': metaprompt_dict['index'],
            'metaprompt': 'dummy',
            'prompts': [
                {
                    'index': 0,
                    'content': metaprompt_dict['metaprompt']
                }
            ]
        })

with open(output_json_path, 'w') as f:
    json.dump(json_dict, f)
    