import re
import json
import os
import random
from classes import *
from openai import OpenAI
from tqdm import tqdm

random.seed(42)

# DomainNet
prompt_json_path = './prompts/synthclip_birds31.json'
dataset_count = birds31_count; imagenet=False
NUM_PROMPTS = 50

def generate_img_caption(client, cls, imagenet=False):
    if imagenet:
        cls_tmp = ImageNet_description[cls]
    else: 
        cls_tmp = cls.replace('_',' ')
    base_message = f"Your task is to write me an image caption that includes and visually describes a scene around a concept. Your concept is {cls_tmp}. Output one single grammatically correct caption that is no longer than 15 words. Do not output any notes, word counts, facts, etc. Output one single sentence only."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_message}
        ]
    )
    response_content = response.choices[0].message.content
    return response_content # assume work well w/ condition: Output one single sentence only.
    

client = OpenAI()

if os.path.exists(prompt_json_path):
    with open(prompt_json_path, 'r') as f:
        totalprompt_dict = json.load(f)
else:
    totalprompt_dict = {}

for cls in tqdm(dataset_count):
    if cls in totalprompt_dict:
        print(f"Skipping {cls}")
        continue
    totalprompt_dict[cls] = {'metaprompts': []}
    class_prompt_list = []
    
    for i in tqdm(range(NUM_PROMPTS)):
        img_caption = generate_img_caption(client, cls, imagenet=imagenet)
        
        totalprompt_dict[cls]['metaprompts'].append(
            {
                'index': i,
                'metaprompt': 'dummy',
                'prompts': [
                    {
                        'index': 0,
                        'content': img_caption
                    }
                ]
            }
        )

    with open(prompt_json_path, 'w') as f:
        json.dump(totalprompt_dict, f)
