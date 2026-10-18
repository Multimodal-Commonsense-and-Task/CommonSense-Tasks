import re
import os
import json
import random
from classes import *
from openai import OpenAI
from tqdm import tqdm
from copy import deepcopy

# original, new1, new2
# base_message = f"To generate an image using a text-to-generation model, I created a prompt based on the concept. However, some sentences have contexts that do not fit the concept, so I would like to improve them into sentences with appropriate contexts. For example, when given the sentence 'A photo of a whale in the football field,' this has a context that does not fit the concept of a football field. This can be improved to 'A photo of a whale in the ocean.' When given a concept and the corresponding prompt, please refine it into a sentence with an appropriate context.\nconcept:{concept}\nprompt:{prompt}\nOutput one single grammatically correct caption with refinement. Do not output any notes, word counts, facts, etc. Output one single sentence only."
# base_message = f"To generate an image using a text-to-generation model, I created a prompt based on the concept. However, some sentences have contexts that do not fit the concept, so I would like to improve them into sentences with appropriate contexts. For example, when given the sentence 'A sunset beach scene with an elephant in warm, pastel colors.' this has a context(beach) that does not fit the concept of an elephant. This can be improved to 'A sunset savanna scene with an elephant in warm, pastel colors.' When given a concept and the corresponding prompt, please refine it into a sentence with an appropriate context.\nconcept:{concept_tmp}\nprompt:{prompt}\nOutput one single grammatically correct caption with refinement. Do not output any notes, word counts, facts, etc. Output one single sentence only."
# base_message = f"To generate an image using a text-to-generation model, I created a prompt based on the concept. However, some sentences have contexts that do not fit the concept, so I would like to improve them into sentences with appropriate contexts. For example, when given the sentence 'A surreal underwater world featuring giraffes in pastel colors.' this has a context(underwater world) that does not fit the concept of a giraffe. This can be improved to 'A surreal savanna landscape featuring giraffes in pastel colors.' When given a concept and the corresponding prompt, please refine it into a sentence with an appropriate context.\nconcept:{concept_tmp}\nprompt:{prompt}\nOutput one single grammatically correct caption with refinement. Do not output any notes, word counts, facts, etc. Output one single sentence only."

random.seed(42)

def generate_refined_caption(client, concept, prompt):
    concept_tmp = concept.replace('_',' ')
    base_message = f"To generate an image using a text-to-generation model, I created a prompt based on the concept. However, some sentences have contexts that do not fit the concept, so I would like to improve them into sentences with appropriate contexts. For example, when given the sentence 'A photo of a whale in the football field,' this has a context that does not fit the concept of a football field. This can be improved to 'A photo of a whale in the ocean.' When given a concept and the corresponding prompt, please refine it into a sentence with an appropriate context.\nconcept:{concept}\nprompt:{prompt}\nOutput one single grammatically correct caption with refinement. Do not output any notes, word counts, facts, etc. Output one single sentence only."
    
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
# important
static_prompt_json_path = './prompts/gpt4_hierarchy_cot_50_2.json'
refined_prompt_json_path = './prompts/gpt4_hierarchy_cot_50_2_refined_DomainNet.json'
dataset_count = DomainNet_count
# important

# Load static prompt
with open(static_prompt_json_path, 'r') as f:
    static_prompt = json.load(f)

# Convert static prompt to dynamic prompt
dynamic_prompt_dict = {}

# Copy metaprompts
for cls in dataset_count.keys():
    dynamic_prompt_dict[cls] = {
        "metaprompts": []
    }
    for metaprompt_dict in static_prompt['metaprompts']:
        dynamic_prompt_dict[cls]['metaprompts'].append({
            "index": metaprompt_dict['index'],
            "metaprompt": metaprompt_dict['metaprompt'],
            "prompts": [
                {
                    "index": prompt_dict['index'],
                    "content": prompt_dict['content'].replace("[concept]", cls.replace('_', ' '))
                } for prompt_dict in metaprompt_dict['prompts']
                
            ]
        })

if os.path.exists(refined_prompt_json_path):
    with open(refined_prompt_json_path, 'r') as f:
        refined_prompt_dict = json.load(f)
else:
    refined_prompt_dict = {}
    
# Generate refined prompt
for cls, cls_prompt_dict in tqdm(dynamic_prompt_dict.items()):
    if cls in refined_prompt_dict.keys():
        print(f"{cls} already exists in refined prompt.")
        continue
    cls_prompt_list = cls_prompt_dict['metaprompts']
    for metaprompt_dict in cls_prompt_list:
        for prompt_dict in metaprompt_dict['prompts']:
            print(f"Before: {prompt_dict['content']}")
            new_prompt = generate_refined_caption(client, cls, prompt_dict['content'])
            prompt_dict['content'] = new_prompt
            print(f"After: {prompt_dict['content']}\n")
            
    refined_prompt_dict[cls] = cls_prompt_dict
    with open(refined_prompt_json_path, 'w') as f:
        json.dump(refined_prompt_dict, f)
        