import re
import json
from openai import OpenAI
from tqdm import tqdm
from classes import *
import random

# calculation for pricing
# one stage: $ 0.0102 / class (171 input tokens in aver., 11 output tokens)
# two stage: $ 0.052 / class (166 input tokens in aver., 14 output tokens)

# total: $ 0.062 / class

# two parts in this code!
# important
dataset_count = PACS_count
metaprompt_json_path = './prompts/temp_base_metaprompts_dynamic.json' # First stage result
totalprompt_json_path = './prompts/gpt4_hierarchy_cot_dynamic_PACS.json' # Second stage result
num_metaprompts = 10
num_prompts_per_metaprompt = 5
# important

def generate_prompt_stage1(client, cls, previous_prompt_list):
    cls_tmp = cls.replace('_', ' ')
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt of a specific concept. Your concept is {cls_tmp}. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
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
            # {"role": "assistant", "content": "prompt: A photo of [concept] in neutral tones"},
            # {"role": "user", "content": base_message}
        ]
    )

    response_content = response.choices[0].message.content
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

def generate_prompt_stage2(client, cls, metaprompt, previous_prompt_list):
    cls_tmp = cls.replace('_',' ')
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Rewrite '{metaprompt}' and the rewritten prompt should be similar but slightly different to '{metaprompt}'. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    
    end_message = f"\nPlease create one prompt sentence (under 15 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '{cls_tmp}'.\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message},
            # {"role": "assistant", "content": "prompt: A photo of [concept] in neutral tones"},
            # {"role": "user", "content": base_message}
        ]
    )

    response_content = response.choices[0].message.content
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

client = OpenAI()


# # For the first stage: metaprompts
# metaprompt_dict = {}
# for cls in tqdm(dataset_count):
#     cls_tmp = cls.replace('_',' ')
#     metaprompts = [f'A photo of a {cls_tmp}.']
#     for i in tqdm(range(num_metaprompts-1)):
#         try:
#             prompt = generate_prompt_stage1(client, cls, metaprompts)
#             print(f"Previous prompt list: {metaprompts}")
#             print(f"Generated metaprompt for stage: {prompt}")
#             metaprompts.append(prompt)
#         except Exception as e:
#             print(e)
#             pass
        
#     metaprompt_dict[cls]=metaprompts

# with open(metaprompt_json_path,'w') as f:
#     json.dump(metaprompt_dict, f)

# For the second stage: specfic prompts (uncomment below)
# Diversified prompt generation

with open(metaprompt_json_path, 'r') as f:
    metaprompt_dict = json.load(f)
    
totalprompt_dict = {}
for cls in tqdm(dataset_count):
    totalprompt_dict[cls] = {'metaprompts': []}
    class_prompt_list = []
    
    metaprompts = metaprompt_dict[cls]
    
    cnt = 0
    for i, metaprompt in enumerate(metaprompts):
        diversified_prompts=[]
        for j in range(num_prompts_per_metaprompt):
            try:
                prompt = generate_prompt_stage2(client, cls, metaprompt, diversified_prompts)
                print(f"previous generated prompts: {diversified_prompts}")
                print(f"Generated prompt: {prompt}")
                diversified_prompts.append(prompt)
                
                totalprompt_dict[cls]['metaprompts'].append(
            {
                'index': cnt,
                'metaprompt': metaprompt,
                'prompts': [
                    {
                        'index': 0,
                        'content': prompt
                    }
                ]
            }
        )
                cnt+=1
            except Exception as e:
                print(e)
                pass

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)