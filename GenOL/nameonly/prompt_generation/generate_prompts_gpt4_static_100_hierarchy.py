import re
import json
import random
from openai import OpenAI
from tqdm import tqdm

# Prompt format에 맞도록 수정해야 함 (0516)

def generate_prompt_stage1(client, previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    end_message = f"\nPlease create one prompt sentence (under 10 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept]'.\n"
    
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

def generate_prompt_stage2(client, previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    
    end_message = f"\nPlease create one prompt sentence (under 15 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept]'.\n"
    
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

num_metaprompts = 14
num_prompts_per_metaprompt = 14
max_prompts = 200
totalprompt_json_path = "prompts/gpt4_hierarchy_cot_200.json"
client = OpenAI()


# you should choose 3 proper metaprompts!
totalprompts = []
metaprompts = ['A photo of a [concept].']

for i in tqdm(range(num_metaprompts - 1)): # hard coded: 3 examples
    try:
        prompt = generate_prompt_stage1(client, metaprompts)
        print(f"Previous prompt list: {metaprompts}")
        print(f"Generated metaprompt for stage: {prompt}")
        metaprompts.append(prompt)
    except Exception as e:
        print(e)
        pass
totalprompts.extend(metaprompts)
metaprompts_before = metaprompts

# Second stage
metaprompts = []
for i, metaprompt in enumerate(tqdm(metaprompts_before)):
    cot_list = [metaprompt]
    
    for j in range(1, num_prompts_per_metaprompt + 1): # hard-coded: 3 examples
        while True:
            try:
                prompt = generate_prompt_stage2(client, cot_list)
                print(f"previous generated prompts: {cot_list}")
                print(f"Generated prompt: {prompt}")
                assert '[concept]' in prompt
                metaprompts.append(prompt)
                cot_list.append(prompt)
                break
            except Exception as e:
                print(e)
                pass
totalprompts.extend(metaprompts)
metaprompts_before = metaprompts

# Save after the second stage
final_prompt_list = random.sample(totalprompts, max_prompts)

# Generate final json
totalprompt_dict = {'metaprompts': []}
totalprompt_dict['metaprompts'] = [
    {
        'index': 0,
        'metaprompt': 'dummy',
        'prompts': [
            {
                'index': i,
                'content': prompt
            } for i, prompt in enumerate(final_prompt_list)
        ]
    }
]

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)

# Third stage
metaprompts = []
for i, metaprompt in enumerate(tqdm(metaprompts_before)):
    cot_list = [metaprompt]
    
    for j in range(1, num_prompts_per_metaprompt + 1): # hard-coded: 3 examples
        while True:
            try:
                prompt = generate_prompt_stage2(client, cot_list)
                print(f"previous generated prompts: {cot_list}")
                print(f"Generated prompt: {prompt}")
                assert '[concept]' in prompt
                metaprompts.append(prompt)
                cot_list.append(prompt)
                break
            except Exception as e:
                print(e)
                pass
totalprompts.extend(metaprompts)
metaprompts_before = metaprompts

# # Save after 3 stages
# final_prompt_list = random.sample(totalprompts, max_prompts)

# # Generate final json
# totalprompt_dict = {'metaprompts': []}
# totalprompt_dict['metaprompts'] = [
#     {
#         'index': 0,
#         'metaprompt': 'dummy',
#         'prompts': [
#             {
#                 'index': i,
#                 'content': prompt
#             } for i, prompt in enumerate(final_prompt_list)
#         ]
#     }
# ]

# with open(totalprompt_json_path, 'w') as f:
#     json.dump(totalprompt_dict, f)

# Fourth stage
metaprompts = []
for i, metaprompt in enumerate(tqdm(metaprompts_before)):
    cot_list = [metaprompt]
    
    for j in range(1, num_prompts_per_metaprompt + 1): # hard-coded: 3 examples
        while True:
            try:
                prompt = generate_prompt_stage2(client, cot_list)
                print(f"previous generated prompts: {cot_list}")
                print(f"Generated prompt: {prompt}")
                assert '[concept]' in prompt
                metaprompts.append(prompt)
                cot_list.append(prompt)
                break
            except Exception as e:
                print(e)
                pass
totalprompts.extend(metaprompts)
metaprompts_before = metaprompts

# Save after 4 stages
final_prompt_list = random.sample(totalprompts, max_prompts)

# Generate final json
totalprompt_dict = {'metaprompts': []}
totalprompt_dict['metaprompts'] = [
    {
        'index': 0,
        'metaprompt': 'dummy',
        'prompts': [
            {
                'index': i,
                'content': prompt
            } for i, prompt in enumerate(final_prompt_list)
        ]
    }
]

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)

breakpoint()
# Fifth stage
metaprompts = []
for i, metaprompt in enumerate(tqdm(metaprompts_before)):
    cot_list = [metaprompt]
    
    for j in range(1, num_prompts_per_metaprompt + 1): # hard-coded: 3 examples
        while True:
            try:
                prompt = generate_prompt_stage2(client, cot_list)
                print(f"previous generated prompts: {cot_list}")
                print(f"Generated prompt: {prompt}")
                assert '[concept]' in prompt
                metaprompts.append(prompt)
                cot_list.append(prompt)
                break
            except Exception as e:
                print(e)
                pass
totalprompts.extend(metaprompts)
metaprompts_before = metaprompts
breakpoint()
final_prompt_list = random.sample(totalprompts, max_prompts)

# Generate final json
totalprompt_dict = {'metaprompts': []}
totalprompt_dict['metaprompts'] = [
    {
        'index': 0,
        'metaprompt': 'dummy',
        'prompts': [
            {
                'index': i,
                'content': prompt
            } for i, prompt in enumerate(final_prompt_list)
        ]
    }
]

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
