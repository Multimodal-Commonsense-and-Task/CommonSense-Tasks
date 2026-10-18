import json
import pickle
import numpy as np
def get_object(train_data):
    seen_action_object = []
    for data in train_data:
        if (data["action_class"], data["object_class"]) not in seen_action_object:
            seen_action_object.append((data["action_class"], data["object_class"]))
    return seen_action_object

with open(file=f'collections/Bongard-HOI/ma_splits/Bongard-HOI_split_record.pkl', mode='rb') as f:
    split_config = pickle.load(f)

with open(f"collections/Bongard-HOI/ma/7_set/Bongard-HOI_train_seed1.json") as fp:
    train_datalists = json.load(fp)
    

print(split_config[1]["train_eval_point"])

split_config[1]["train_eval_point"] = np.array(split_config[1]["train_eval_point"]) * 2
for i in range(len(split_config[1]["train_eval_point"])): 
    start = sum(split_config[1]["train_eval_point"][:i])
    end = sum(split_config[1]["train_eval_point"][:i+1])
    print(start, end)
    print(len(get_object(train_datalists[start:end])))
    print(get_object(train_datalists[start:end]))
