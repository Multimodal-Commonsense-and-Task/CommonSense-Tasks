import re
import pickle
import json
from openai import OpenAI
from tqdm import tqdm
from classes import *

def generate_prompt_noname(client, concept_name):
    message = f"""Q: What are useful visual features for distinguishing a lemur in a photo?
    A: There are several useful visual features to tell there is a lemur in a photo:
    - four-limbed primate
    - black, grey, white, brown, or red-brown
    - wet and hairless nose with curved nostrils
    - long tail
    - large eyes
    - furry bodies
    - clawed hands and feet
    Q: What are useful visual features for distinguishing a television in a photo?
    A: There are several useful visual features to tell there is a television in a photo:
    - electronic device
    - black or grey
    - a large, rectangular screen
    - a stand or mount to support the screen
    - one or more speakers
    - a power cord
    - input ports for connecting to other devices
    - a remote control
    Q: What are useful features for distinguishing a {concept_name} in a photo?
    A: There are several useful visual features to tell there is a {concept_name} in a photo:
    - 
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ]
    )
    while True:
        response_content = response.choices[0].message.content
        response_list = response_content.split('\n')
        response_processed_list = [item.strip(' -') for item in response_list]
        if len(response_processed_list) >= 3:
            print(f"Response list: {response_processed_list}")
            return response_processed_list
        else:
            print(f"Response too short")
            print(f"Response: {response_processed_list}")


cls_count_dict = DomainNet_count
result_json_path = './prompts/noname_PACS.json'

client = OpenAI()
classes = list(cls_count_dict.keys())

totalprompt_dict = {}
for cls in tqdm(classes):
    responses = generate_prompt_noname(client, cls)
    totalprompt_dict[cls] = responses

with open(result_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)