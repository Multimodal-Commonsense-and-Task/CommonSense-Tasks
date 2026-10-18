import re
import pickle
import json
from openai import OpenAI
from tqdm import tqdm
from classes import *

# Prompt format에 맞도록 수정해야 함 (0516)
# wo_cot에서 stage 1에서는 그냥 ui에서 만드는거랑 똑같다.
# stage 2에서는 stage 2의 metaprompt마다 slightly different하게 만드는 것.


def generate_prompt_stage2(client, metaprompt, num_prompts):
    base_message = f"To generate images using a text-to-image generation model, I need to create {num_prompts} prompts. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Rewrite '{metaprompt}' and the rewritten prompt should be similar to '{metaprompt}'. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept].\n"

    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message},
            {"role": "assistant", "content": "prompt: A photo of [concept] in neutral tones"},
            {"role": "user", "content": base_message}
        ]
    )

    response_content = response.choices[0].message.content
    breakpoint()
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

metaprompt_json_path = './prompts/temp_base_metaprompts.json'
totalprompt_json_path = './prompts/gpt4_wo_cot.json'

client = OpenAI()
num_metaprompts = 10
num_prompts_per_metaprompt = 5


# Diversified prompt generation
# Load metaprompt pickle file
with open(metaprompt_json_path, 'r') as f:
    metaprompt_list = json.load(f)

totalprompt_dict = {'metaprompts': []}
for i, metaprompt in enumerate(tqdm(metaprompt_list)):
    metaprompt_dict = {'index': i, 'metaprompt': metaprompt, 'prompts': []}
    diversified_prompts = []
    for j in range(num_prompts_per_metaprompt):
        try:
            prompt = generate_prompt_stage2(client, metaprompt, num_prompts=num_prompts_per_metaprompt)
            print(f"previous generated prompts: {diversified_prompts}")
            print(f"Generated prompt: {prompt}")
            diversified_prompts.append(prompt)
            metaprompt_dict['prompts'].append({'index': j, 'content': prompt})
        except Exception as e:
            print(e)

    totalprompt_dict['metaprompts'].append(metaprompt_dict)

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
