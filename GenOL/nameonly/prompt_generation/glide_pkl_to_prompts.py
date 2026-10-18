import pickle
import json

ma_json_path = '../../collections/Bongard-HOI/ma_splits/Bongard-HOI_train_seed1.json'
LE_pickle_path = './prompts/ma_split_train_seed1.pkl'
result_prompt_path = './prompts/generated_LE_ver1.json'

ma_dataset = json.load(open(ma_json_path, 'r'))
pkl_dataset = pickle.load(open(LE_pickle_path, 'rb'))
result_list = []

for dataset_dict in ma_dataset:
    result_dict = {}
    result_dict['id'] = dataset_dict['id']
    result_dict['object_class'] = dataset_dict['object_class']
    result_dict['action_class'] = dataset_dict['action_class']
    
    # Get pos/neg prompts from pkl
    prompt_list = pkl_dataset[str(dataset_dict['id'])]
    positive_prompts = [prompt_list[0]] * (len(prompt_list) - 1)
    negative_prompts = prompt_list[1:]
    assert len(positive_prompts) == len(negative_prompts)
    
    # Assign pos/neg prompts to result_dict
    result_dict['positive_prompts'] = positive_prompts
    result_dict['negative_prompts'] = negative_prompts

    result_list.append(result_dict)
    
with open(result_prompt_path, 'w') as f:
    json.dump(result_list, f)
    