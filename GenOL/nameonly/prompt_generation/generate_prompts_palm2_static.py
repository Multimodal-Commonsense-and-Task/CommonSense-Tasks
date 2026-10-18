import os
import re
import pickle
import json
from openai import OpenAI
from tqdm import tqdm
import google.generativeai as palm 
palm.configure(api_key=os.environ["GOOGLE_API_KEY"])

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name

def generate_prompt_stage1(previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these for the sake of diversity."
    end_message = f"\nPlease create one prompt sentence (under 10 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept].\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=base_message,
        temperature=1.0,
        # The maximum length of the response
        max_output_tokens=800,
    )

    response_content = completion.result
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

def generate_prompt_stage2(metaprompt, previous_prompt_list):
    base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. Keep the domain photorealistic and use different visual scenes and visual styles or different color profiles/ palettes. The prompt should be similar to '{metaprompt}' but slightly different. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these for the sake of diversity."
    
    end_message = f"\nPlease create one prompt sentence (under 15 words) that fits this description. Please ensure the response format is strictly 'prompt: answer' and include the word '[concept].\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=base_message,
        temperature=0.5,
        # The maximum length of the response
        max_output_tokens=800,
    )

    response_content = completion.result
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

metaprompt_json_path = './prompts/palm2_temp.json'
totalprompt_json_path = './prompts/palm2_static_totalprompts_with_cot_50_photorealistic.json'

num_metaprompts = 10
num_prompts_per_metaprompt = 5

# # Metaprompt generation
# metaprompts = ['A photo of a [concept].']
# for i in tqdm(range(num_metaprompts - 1)):
#     try:
#         prompt = generate_prompt_stage1(metaprompts)
#         print(f"Previous prompt list: {metaprompts}")
#         print(f"Generated metaprompt for stage: {prompt}")
#         metaprompts.append(prompt)
#     except:
#         pass

# with open(metaprompt_json_path, 'w') as f:
#     json.dump(metaprompts, f)


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
            prompt = generate_prompt_stage2(metaprompt, diversified_prompts)
            print(f"previous generated prompts: {diversified_prompts}")
            print(f"Generated prompt: {prompt}")
            diversified_prompts.append(prompt)
            metaprompt_dict['prompts'].append({'index': j, 'content': prompt})
        except:
            print(f"Error while generating diversified prompts for {metaprompt}")
    totalprompt_dict['metaprompts'].append(metaprompt_dict)

with open(totalprompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
