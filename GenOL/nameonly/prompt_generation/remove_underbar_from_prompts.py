import json

prompt_path = './prompts/LE_DomainNet.json'
new_prompt_path = './prompts/LE_DomainNet_new.json'
with open(prompt_path, 'r') as f:
    prompt_dict = json.load(f)
    
if 'metaprompts' in prompt_dict:
    metaprompt_list = prompt_dict['metaprompts']
    for metaprompt_dict in metaprompt_list:
        prompt_list = metaprompt_dict['prompts']
        for innerprompt_dict in prompt_list:
            innerprompt_dict['content'] = innerprompt_dict['content'].replace('_', ' ')
            
else:
    for cls in prompt_dict:
        metaprompt_list = prompt_dict[cls]['metaprompts']
        for metaprompt_dict in metaprompt_list:
            prompt_list = metaprompt_dict['prompts']
            for innerprompt_dict in prompt_list:
                innerprompt_dict['content'] = innerprompt_dict['content'].replace('_', ' ')

with open(new_prompt_path, 'w') as f:
    json.dump(prompt_dict, f)