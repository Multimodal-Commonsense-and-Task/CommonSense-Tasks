import re
import json
import ast
import os
from classes import *
from openai import OpenAI
from tqdm import tqdm

count_dict = birds31_count; imagenet=False
NUM_BACKGROUNDS = 100
backgrounds_json_path = './prompts/backgrounds_synclr_birds31.json'
class_list = list(count_dict.keys())
client = OpenAI()

def get_backgrounds(client, cls, num_backgrounds, imagenet=False):
    while True:
        try:
            if imagenet:
                cls_name = ImageNet_description[cls]
            else:
                cls_name = cls.replace('_', ' ')
            prompt = f"""To generate images from various backgrounds, I want to create a list of backgrounds suitable for a specific class. Background should be suitable for the class. For example, a football field background would not be suitable for a class like blue whale. Please generate a list of {num_backgrounds} backgrounds for the following class.\nclass: {cls_name}\nPlease make sure that the response format is in the form of backgrounds: ["background1", "background2", ...], and the length of the list should be {num_backgrounds}. Do not output any unnecessary information.\n"""
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"\[.*\]", response_content)
            result_list = ast.literal_eval(match.group())
        except Exception as e:
            print(e)
            print(f"Wrong response content: {response_content}")
            continue
        
        if len(result_list) >= 0.5 * num_backgrounds:
            break

    return result_list

if os.path.exists(backgrounds_json_path):
    with open(backgrounds_json_path, 'r') as f:
        result_dict = json.load(f)
else:
    result_dict = {}
    
for cls in tqdm(class_list):
    if cls in result_dict:
        print(f"Skipping class {cls}")
        continue
    backgrounds = []
    while len(backgrounds) < NUM_BACKGROUNDS:
        print(f"Getting {NUM_BACKGROUNDS - len(backgrounds)} backgrounds for class {cls}")
        response_list = get_backgrounds(client, cls, NUM_BACKGROUNDS - len(backgrounds),
                                        imagenet=imagenet)
        backgrounds.extend(response_list)
    result_dict[cls] = backgrounds[:NUM_BACKGROUNDS]

    with open(backgrounds_json_path, 'w') as f:
        json.dump(result_dict, f)
