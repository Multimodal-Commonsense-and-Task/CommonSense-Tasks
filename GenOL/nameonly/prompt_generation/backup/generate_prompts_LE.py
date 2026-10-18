import re
import json
import nltk
import random
from classes import *
from tqdm import tqdm
import pickle

is_NICO = True # NICO.pkl has cls name without underbar
prompt_json_path = './prompts/LE_NICO.json'
dataset_count = NICO_count
file_path = '../../LE_prompts/prompt_LE_NICO.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

totalprompt_dict = {}
for cls in tqdm(dataset_count):
    totalprompt_dict[cls] = {'metaprompts': []}
    class_prompt_list = []
    
    for i in range(50):
        if is_NICO:
            cls_without = cls.replace('_',' ')
            img_caption = data[cls_without][i]
        else:
            img_caption = data[cls][i]
        
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
