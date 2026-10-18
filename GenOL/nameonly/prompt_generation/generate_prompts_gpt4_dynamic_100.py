import re
import json
import random
from openai import OpenAI
from tqdm import tqdm
from classes import *
import os

# dynamic, class-specific
def generate_prompt_stage1(client, cls, previous_prompt_list):
    cls_tmp = cls.replace('_',' ')
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    end_message = f"\nPlease create one prompt sentence (under 10 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '{cls_tmp}'.\n"
    
    # For CoT Prompting
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message},
        ]
    )
    response_content = response.choices[0].message.content
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

def generate_prompt_stage2(client, cls, previous_prompt_list):
    cls_tmp = cls.replace('_',' ')
    
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    end_message = f"\nPlease create one prompt sentence (under 15 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '{cls_tmp}'.\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message
    
    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message}
        ]
    )
    response_content = response.choices[0].message.content
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

metaprompt_json_path = './prompts/temp_base_metaprompts_dynamic_7_PACS.json' # First stage result
totalprompt_json_path = './prompts/gpt4_hierarchy_cot_dynamic_50_new_PACS.json' # Second stage result
num_metaprompts = 7
num_prompts_per_metaprompt = 7
max_prompts = 50
dataset_count = PACS_count

client = OpenAI()

# # For the first stage
# metaprompt_dict = {}
# for cls in tqdm(dataset_count):
#     cls_tmp = cls.replace('_',' ')
    
#     # use template of 2nd version (50_2, 100_2)
#     metaprompts = [f'A photo of a {cls_tmp}.',f'A detailed sketch of {cls_tmp}.', f'A minimalist illustration of {cls_tmp}.']
#     for i in tqdm(range(num_metaprompts-3)):
#         try:
#             prompt = generate_prompt_stage1(client, cls, metaprompts)
#             print(f"Previous prompt list: {metaprompts}")
#             print(f"Generated metaprompt for stage: {prompt}")
#             metaprompts.append(prompt)
#         except Exception as e:
#             print(e)
#             pass
#     metaprompt_dict[cls] = metaprompts

# with open(metaprompt_json_path, 'w') as f:
#     json.dump(metaprompt_dict, f)


# For the second stage (uncomment below)
with open(metaprompt_json_path, 'r') as f:
    metaprompt_dict = json.load(f)

if os.path.exists(totalprompt_json_path):
    with open(totalprompt_json_path, 'r') as f:
        totalprompt_dict = json.load(f)
else:
    totalprompt_dict = {}

for cls in tqdm(dataset_count):
    if cls in totalprompt_dict:
        print(f"Pass: {cls}")
        continue
    cls_tmp = cls.replace('_',' ')
    prompt_list_tmp = []
    totalprompt_dict[cls] = {'metaprompts': []}
    
    metaprompt_list = metaprompt_dict[cls] 
    
    for i, metaprompt in enumerate(tqdm(metaprompt_list)):
        cot_list = [metaprompt]
        
        tmp = [x for x in range(len(metaprompt_list)) if x!=i]
        sampled_numbers = random.sample(tmp,2)
        
        for n in sampled_numbers:
            cot_list.append(metaprompt_list[n]) # only append at cot_list
            
        for j in range(1, num_prompts_per_metaprompt+1):
            while True:
                try:
                    prompt = generate_prompt_stage2(client,cls,cot_list)
                    print(f"previous generated prompts: {cot_list}")
                    print(f"Generated prompt: {prompt}")
                    prompt_list_tmp.append(prompt)
                    cot_list.append(prompt)
                    break
                except Exception as e:
                    print(e)
                    pass
                
    final_prompt_list = metaprompt_list + random.sample(prompt_list_tmp, max_prompts - num_metaprompts)
    
    for i in range(max_prompts):
        try:
            totalprompt_dict[cls]['metaprompts'].append(
            {
                'index': i,
                'metaprompt': 'dummy',
                'prompts': [
                    {
                        'index': 0,
                        'content': final_prompt_list[i]
                    }
                ]
            }
        )
            
        except Exception as e:
            print(e)
            pass
    
    with open(totalprompt_json_path, 'w') as f:
        json.dump(totalprompt_dict, f)
        
