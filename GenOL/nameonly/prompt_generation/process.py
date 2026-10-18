# import json

# prompt_path = './prompts/dynamic_totalprompts_PACS.json'
# with open(prompt_path, 'r') as f:
#     prompt_dict = json.load(f)

# static_prompt_dict = {}
# for cls, metaprompt_list in prompt_dict.items():
#     cls_dict = {'metaprompts': []}
#     for prompt_dict in metaprompt_list:
#         metaprompt_dict = {
#             'index': prompt_dict['index'],
#             'metaprompt': prompt_dict['metaprompt'],
#             'prompts': prompt_dict['prompts']
#         }
#         cls_dict['metaprompts'].append(metaprompt_dict)

#     static_prompt_dict[cls] = cls_dict


# # for i, (k, v) in enumerate(prompt_dict.items()):
# #     metaprompt_dict = {
# #         'index': i,
# #         'metaprompt': k,
# #         'prompts': v['prompts']
# #     }
# #     static_prompt_dict['metaprompts'].append(metaprompt_dict)

# with open('./prompts/dynamic_totalpromptsprocessed.json', 'w') as f:
#     json.dump(static_prompt_dict, f)


# 쭉 써놓은 CoT 없는 meta - diversified를 json으로 바꾸기
import json

metaprompt_num = 10
diversified_num = 5

txt_path = './prompts/prompts.txt'
json_path = './prompts/base_prompts_no_cot.json'

with open(txt_path, 'r') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

json_dict = {'metaprompts': []}
for i in range(metaprompt_num):
    metaprompt_dict = {}
    metaprompt_idx = i * (diversified_num + 1)
    metaprompt_dict['index'] = i
    metaprompt_dict['metaprompt'] = lines[metaprompt_idx]
    metaprompt_dict['prompts'] = []
    
    for j in range(diversified_num):
        diversified_idx = metaprompt_idx + j + 1
        metaprompt_dict['prompts'].append({
            'index': j,
            'content': lines[diversified_idx]
        })
    
    json_dict['metaprompts'].append(metaprompt_dict)

with open(json_path, 'w') as f:
    json.dump(json_dict, f)
