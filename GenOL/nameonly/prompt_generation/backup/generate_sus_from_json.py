import json

input_path = './prompts/sus_PACS_before_process.json'
output_path = './prompts/sus_PACS.json'

with open(input_path, 'r') as f:
    json_dict = json.load(f)

result_dict = {}
for cls, prompts_list in json_dict.items():
    result_dict[cls] = {'metaprompts': []}
    for i, prompt in enumerate(prompts_list):
        result_dict[cls]['metaprompts'].append({
            'index': i,
            'metaprompt': 'dummy',
            'prompts': [
                {
                    'index': 0,
                    'content': prompt
                }
            ]
        })

with open(output_path, 'w') as f:
    json.dump(result_dict, f)