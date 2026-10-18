import os
import re
import pickle
import json
from tqdm import tqdm
from classes import *
import google.generativeai as palm 
palm.configure(api_key=os.environ["GOOGLE_API_KEY"])

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name

def generate_prompt_stage1(previous_prompt_list, concept_name):
    if len(previous_prompt_list) == 0:
        base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. The images will be used to train a classifier, so they should feature various visual scenes and styles, or different color profiles/palettes. The subject of the image is '{concept_name}'."
    else:
        base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. The images will be used to train a classifier, so they should feature various visual scenes and styles, or different color profiles/palettes. The subject of the image is '{concept_name}'. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap (style, color, etc.) with these for the sake of diversity."
    
    end_message = f"\nPlease create one prompt sentence (under 20 words) that fits this description. Please ensure the response format is strictly 'prompt: answer'.\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=base_message,
        temperature=0,
        # The maximum length of the response
        max_output_tokens=800,
    )
    return completion.result[8:]


def generate_prompt_stage2(metaprompt, previous_prompt_list, concept_name):
    if len(previous_prompt_list) == 0:
        base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. The prompt should be similar to '{metaprompt}' but slightly different. The subject of the image is '{concept_name}'."
    else:
        base_message = f"To generate images using a text-to-image generation model, I need to create a prompt. The prompt should be similar to '{metaprompt}' but slightly different. The subject of the image is '{concept_name}'. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap (style, color, etc.) with these for the sake of diversity."
    
    end_message = f"\nPlease create one prompt sentence (under 30 words) that fits this description. Please ensure the response format is strictly 'prompt: answer'.\n"
    
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message

    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=base_message,
        temperature=0,
        # The maximum length of the response
        max_output_tokens=800,
    )
    return completion.result[8:]

cls_count_dict = PACS_count
metaprompt_json_path = './prompts/palm2_metaprompts_PACS.json'
totalprompt_json_path = './prompts/palm2_totalprompts_PACS.json'

classes = list(cls_count_dict.keys())
num_metaprompts = 10
num_prompts_per_metaprompt = 5


# -------------------------------------------------------------------
# Metaprompt generation
cls_metaprompt_dict = {}
for cls in tqdm(classes):
    metaprompts = []

    for i in tqdm(range(num_metaprompts), desc=f"Generating metaprompts for {cls}"):
        # try:
        prompt = generate_prompt_stage1(metaprompts, cls)
        print(f"Previous prompt list: {metaprompts}")
        print(f"Generated metaprompt for stage: {prompt}")
        metaprompts.append(prompt)
        # except:
        #     pass

    cls_metaprompt_dict[cls] = metaprompts

with open(metaprompt_json_path, 'w') as f:
    json.dump(cls_metaprompt_dict, f)


# -------------------------------------------------------------------
# Diversified prompt generation
# Load metaprompt pickle file
# with open(metaprompt_json_path, 'r') as f:
#     metaprompt_dict = json.load(f)

# totalprompt_dict = {}
# for cls in classes:
#     totalprompt_dict[cls] = {}
#     metaprompts_cls = metaprompt_dict[cls]
#     for metaprompt in tqdm(metaprompts_cls):
#         # Generate diversified prompts for each metaprompt
#         diversified_prompts = []
#         for i in range(num_prompts_per_metaprompt):
#             prompt = generate_prompt_stage2(metaprompt, diversified_prompts, cls)
#             print(f"previous generated prompts: {diversified_prompts}")
#             print(f"Generated prompt: {prompt}")
#             diversified_prompts.append(prompt)
#         totalprompt_dict[cls][metaprompt] = diversified_prompts

# with open(totalprompt_json_path, 'w') as f:
#     json.dump(totalprompt_dict, f)
