import re
import json
import nltk
from classes import *
from nltk.corpus import wordnet as wn
from openai import OpenAI
from tqdm import tqdm
nltk.download('wordnet')

prompt_json_path = './prompts/fake_DomainNet.json'
dataset_count = DomainNet_count

totalprompt_dict = {}
for cls in dataset_count:
    totalprompt_dict[cls] = {'metaprompts': []}
    synsets = wn.synsets(cls)
    if len(synsets) == 0:
        definition = cls
    else:
        definition = f"{cls}, {synsets[0].definition()}"

    totalprompt_dict[cls]['metaprompts'].append(
        {
            'index': 0,
            'metaprompt': 'dummy',
            'prompts': [
                {
                    'index': 0,
                    'content': definition
                }
            ]
        }
    )

with open(prompt_json_path, 'w') as f:
    json.dump(totalprompt_dict, f)
