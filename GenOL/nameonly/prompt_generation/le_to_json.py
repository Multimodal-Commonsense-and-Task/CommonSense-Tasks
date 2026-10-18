import pickle
import json
import random
from classes import *

dataset_count = birds31_count
pickle_path = './prompts/prompt_LE_bird31.pkl'
json_path = './prompts/LE_birds31.json'
LE_dict = pickle.load(open(pickle_path, 'rb'))
NUM_PROMPTS = 50

result_dict = {}
for cls in dataset_count.keys():
    result_dict[cls] = {
        "metaprompts": [
            {
                "index": 0,
                "metaprompt": "dummy",
                "prompts": []
            }
        ]
    }
    
    prompts = random.sample(LE_dict[cls], NUM_PROMPTS)
    for i, prompt in enumerate(prompts):
        result_dict[cls]["metaprompts"][0]["prompts"].append(
            {
                "index": i,
                "content": prompt
            }
        )

with open(json_path, 'w') as f:
    json.dump(result_dict, f, indent=4)