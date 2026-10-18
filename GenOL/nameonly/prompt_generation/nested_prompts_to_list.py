# Nested prompt structure -> list of prompts

import json
from classes import *

prompt_path = './prompts/palm2_static_totalprompts_with_cot_50.json'
with open(prompt_path, 'r') as f:
    data_dict = json.load(f)

result_list = []
concept_count = 0
for metaprompt_dict in data_dict['metaprompts']:
    for prompt_dict in metaprompt_dict['prompts']:
        result_list.append(prompt_dict['content'])
        if '[concept]' in prompt_dict['content']:
            concept_count += 1

print(result_list)
print(f"Length of list: {len(result_list)}")
print(f"[concept] count: {concept_count}")
# classes = list(NICO_count.keys())
# result_dict = {}
# for i, cls in enumerate(classes):
#     result_dict[cls] = {
#         'metaprompts': [
#             {
#                 'index': i,
#                 'metaprompt': 'dummy',
#                 'prompts': [
#                     {'index': 0, 'content': f"A photo of {cls}"}
#                 ]
#             }
#         ]
#     }

# with open(prompt_path, 'w') as f:
#     json.dump(result_dict, f)