import re
import json
import random
from openai import OpenAI
from tqdm import tqdm

# Prompt format에 맞도록 수정해야 함 (0516)

def generate_prompt_stage1(client, previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    end_message = f"\nPlease create one prompt sentence (under 15 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept]'.\n"
    
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


totalprompt_json_path = './prompts/gpt4_wo_hierarchy_100.json' # Second stage result
num_prompts = 50

client = OpenAI()

prompt_list = ['A photo of a [concept].', 'A colorful vector clipart of [concept].','A simple sketch of [concept] with bold contrasts.']
for i in range(num_prompts - len(prompt_list)):
    try:
        prompt = generate_prompt_stage1(client, prompt_list)
        print(f"Generated prompt: {prompt}")
        assert '[concept]' in prompt, f"Prompt does not contain '[concept]': {prompt}"
        prompt_list.append(prompt)
    except Exception as e:
        print(e)
        pass

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
            } for i, prompt in enumerate(prompt_list)
        ]
    }
]

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
