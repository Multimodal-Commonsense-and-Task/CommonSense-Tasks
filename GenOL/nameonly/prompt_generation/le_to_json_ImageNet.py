import pickle
import json
import random
from classes import *

dataset_count = ImageNet_count
pickle_path = './prompts/prompt_LE_ImageNet.pkl'
json_path = './prompts/LE_ImageNet.json'
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
    cls_name = ImageNet_description[cls]
    prompts = random.sample(LE_dict[cls_name], NUM_PROMPTS)
    for i, prompt in enumerate(prompts):
        result_dict[cls]["metaprompts"][0]["prompts"].append(
            {
                "index": i,
                "content": prompt
            }
        )
    

with open(json_path, 'w') as f:
    json.dump(result_dict, f, indent=4)