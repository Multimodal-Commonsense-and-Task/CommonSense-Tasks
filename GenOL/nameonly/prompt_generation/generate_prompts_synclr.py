import os
import re
import json
import random
from classes import *
from openai import OpenAI
from tqdm import tqdm
from synclr_utils import fg_example, bg_example, fgbg_example, fgrel_example, relation_list

background_json_path = './prompts/backgrounds_synclr_ImageNet.json'; imagenet=True
prompt_json_path = './prompts/synclr_ImageNet_200.json'
dataset_count = ImageNet_count
NUM_PROMPTS = 200

with open(background_json_path, 'r') as f:
    backgrounds_dict = json.load(f)

def sample_prompt(client, cls):
    cls_idx = cls
    if imagenet:
        cls = ImageNet_description[cls]
        
    fg_model_sample = random.random()
    if fg_model_sample < 0.44:
        fg_mode = 0
    elif fg_model_sample < 0.55:
        fg_mode = 1
    else:
        fg_mode = 2
    
    if fg_mode == 0:
        num_example = len(fg_example)
        chosen_idx = random.sample(range(num_example), 3)
        current_prompt = """Generate an image description with an object category:

                            {} => {}

                            {} => {}

                            {} => {}

                            {} =>""".format(
                fg_example[chosen_idx[0]][0], fg_example[chosen_idx[0]][1],
                fg_example[chosen_idx[1]][0], fg_example[chosen_idx[1]][1],
                fg_example[chosen_idx[2]][0], fg_example[chosen_idx[2]][1],
                cls.replace('_', ' '))
    elif fg_mode == 1:
        num_example = len(fgrel_example)
        chosen_idx = random.sample(range(num_example), 3)
        relation = random.choice(relation_list)
        current_prompt = """Generate an image description with an object category and a relation:

                                                                           {}, {} => {}

                                                                           {}, {} => {}

                                                                           {}, {} => {}

                                                                           {}, {} =>""".format(
                        fgrel_example[chosen_idx[0]][0], fgrel_example[chosen_idx[0]][1],
                        fgrel_example[chosen_idx[0]][2],
                        fgrel_example[chosen_idx[1]][0], fgrel_example[chosen_idx[1]][1],
                        fgrel_example[chosen_idx[1]][2],
                        fgrel_example[chosen_idx[2]][0], fgrel_example[chosen_idx[2]][1],
                        fgrel_example[chosen_idx[2]][2],
                        cls.replace('_', ' '), relation)
    else:
        num_example = len(fgbg_example)
        chosen_idx = random.sample(range(num_example), 3)
        background = random.sample(backgrounds_dict[cls_idx], 1)[0]
        current_prompt = """Generate an image description with an object category and an environment category:

                                                    {}, {} => {}

                                                    {}, {} => {}

                                                    {}, {} => {}

                                                    {}, {} =>""".format(
                fgbg_example[chosen_idx[0]][0], fgbg_example[chosen_idx[0]][1], fgbg_example[chosen_idx[0]][2],
                fgbg_example[chosen_idx[1]][0], fgbg_example[chosen_idx[1]][1], fgbg_example[chosen_idx[1]][2],
                fgbg_example[chosen_idx[2]][0], fgbg_example[chosen_idx[2]][1], fgbg_example[chosen_idx[2]][2],
                cls.replace('_', ' '), background)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": current_prompt}
        ]
    )
    response_content = response.choices[0].message.content
    return response_content
    
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
    for i in tqdm(range(NUM_PROMPTS)):
        prompt = sample_prompt(client, cls)

        totalprompt_dict[cls]['metaprompts'].append(
            {
                'index': i,
                'metaprompt': 'dummy',
                'prompts': [
                    {
                        'index': 0,
                        'content': prompt
                    }
                ]
            }
        )

    with open(prompt_json_path, 'w') as f:
        json.dump(totalprompt_dict, f)
