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

first_json_path = './prompts/tree/first.json' # First stage result
second_json_path = './prompts/tree/second.json' # Second stage result
third_json_path = './prompts/tree/third.json' # Third stage result

num_first_prompts = 6
num_second_prompts = 6
num_third_prompts = 6

client = OpenAI()

# # For the first stage
# # you should choose 3 proper metaprompts!
# metaprompts = ['A photo of a [concept].', 'A colorful vector clipart of [concept].','A simple sketch of [concept] with bold contrasts.']

# for i in tqdm(range(num_first_prompts - 3)): # hard coded: 3 examples
#     try:
#         prompt = generate_prompt_stage1(client, metaprompts)
#         print(f"Previous prompt list: {metaprompts}")
#         print(f"Generated metaprompt for stage: {prompt}")
#         metaprompts.append(prompt)
#     except Exception as e:
#         print(e)
#         pass

# with open(first_json_path, 'w') as f:
#     json.dump(metaprompts, f)

# # For the second stage (uncomment below)
# with open(first_json_path, 'r') as f:
#     first_list = json.load(f)

# second_dict = {
#     "metaprompts": []
# }

# for i, metaprompt in enumerate(tqdm(first_list)):
#     cot_list = [metaprompt]
#     prompt_list = []
#     tmp = [x for x in range(len(first_list)) if x!=i]
#     sampled_numbers = random.sample(tmp,2)
    
#     for n in sampled_numbers:
#         cot_list.append(first_list[n]) # only for cot_list, not prompt_list

#     for j in range(1, num_second_prompts + 1): # hard-coded: 3 examples
#         while True:
#             try:
#                 prompt = generate_prompt_stage2(client, cot_list)
#                 print(f"previous generated prompts: {cot_list}")
#                 print(f"Generated prompt: {prompt}")
#                 assert '[concept]' in prompt
#                 prompt_list.append(prompt)
#                 cot_list.append(prompt)
#                 break
#             except Exception as e:
#                 print(e)
#                 pass
#     second_dict['metaprompts'].append({
#         'index': i,
#         'metaprompt_1': metaprompt,
#         'prompts': [
#             {
#                 'index': j,
#                 'content': prompt
#             } for j, prompt in enumerate(prompt_list)
#         ]
#     })

# with open(second_json_path, 'w') as f:
#     json.dump(second_dict, f)

# Third stage (uncomment below)
with open(second_json_path, 'r') as f:
    second_dict = json.load(f)

third_dict = {
    "metaprompts": []
}

for metaprompt_dict in tqdm(second_dict['metaprompts']):
    first_stage_idx = metaprompt_dict['index']
    first_stage_metaprompt = metaprompt_dict['metaprompt_1']
    second_stage_prompts = metaprompt_dict['prompts']
    
    second_stage_metaprompts_dict = {
        'index': first_stage_idx,
        'metaprompt_1': first_stage_metaprompt,
        'prompts': []
    }
    for i, second_stage_prompt in enumerate(second_stage_prompts):
        second_stage_metaprompts_dict['prompts'].append({
            'index': i,
            'metaprompt_2': second_stage_prompt['content'],
            'metaprompts': []
        })
        cot_list = [second_stage_prompt['content']]
        prompt_list = []
        tmp = [x for x in range(len(second_stage_prompts)) if x!=i]
        sampled_numbers = random.sample(tmp,2)
        
        for n in sampled_numbers:
            cot_list.append(second_stage_prompts[n]['content'])
        
        for j in range(1, num_third_prompts + 1): # hard-coded: 3 examples
            while True:
                try:
                    prompt = generate_prompt_stage2(client, cot_list)
                    print(f"previous generated prompts: {cot_list}")
                    print(f"Generated prompt: {prompt}")
                    assert '[concept]' in prompt
                    prompt_list.append(prompt)
                    cot_list.append(prompt)
                    break
                except Exception as e:
                    print(e)
                    pass
        second_stage_metaprompts_dict['prompts'][i]['metaprompts'] = [
            {
                'index': j,
                'content': prompt
            } for j, prompt in enumerate(prompt_list)
        ]
        
    third_dict['metaprompts'].append(second_stage_metaprompts_dict)

with open(third_json_path, 'w') as f:
    json.dump(third_dict, f)
