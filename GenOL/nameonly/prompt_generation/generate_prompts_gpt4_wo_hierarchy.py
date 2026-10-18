import re
import pickle
import json
from openai import OpenAI
from tqdm import tqdm
from classes import *

# Prompt format에 맞도록 수정해야 함 (0516)

def generate_prompt_stage1(client, previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    end_message = f"\nPlease create one prompt sentence (under 15 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept].\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message},
            {"role": "assistant", "content": "prompt: A photo of [concept] in neutral tones."},
            {"role": "user", "content": base_message},
        ]
    )

    response_content = response.choices[0].message.content
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

totalprompt_json_path = './prompts/gpt4_wo_hierarchy_100.json'

client = OpenAI()
num_prompts = 100

prompts = ['A photo of a [concept].']
for i in tqdm(range(num_prompts - 1)):
    try:
        prompt = generate_prompt_stage1(client, prompts)
        print(f"Previous prompt list: {prompts}")
        print(f"Generated metaprompt for stage: {prompt}")
        prompts.append(prompt)
    except:
        pass

totalprompt_dict = {'metaprompts': []}
for i, prompt in enumerate(prompts):
    metaprompt_dict = {
        'index': i,
        'metaprompt': 'dummy',
        'prompts': [
            {
                'index': 0,
                'content': prompt
            }
        ]
    }
    totalprompt_dict['metaprompts'].append(metaprompt_dict)

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
