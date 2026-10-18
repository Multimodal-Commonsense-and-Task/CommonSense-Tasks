# Use action selection when generating negative prompts
import re
import os
import json
import ast
from openai import OpenAI
from tqdm import tqdm

NUM_POSITIVE = 7

def generate_prompt_from_pair(object_class, action_class):
    base_message = f"Using the object-action pair below, please generate a prompt for image generation. For example, if the object is 'sports_ball' and the action is 'kick', the prompt could be 'A photo of a person kicking a sports ball.' If the object is 'orange' and the action is 'hold', the prompt could be 'Holding an orange.' Another example is if the object is 'giraffe' and the action is 'no_interaction', the prompt could be 'A photo of a giraffe.' \n- object: {object_class}\n- action: {action_class}\nPlease write a prompt for image generation. The response format must be strictly prompt: your prompt here.\n"
    while True:
        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "user", "content": base_message},
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"prompt:\s*(.*)", response_content)
            prompt = match.group(1)
            break
        except Exception as e:
            print(e)
            continue
    
    return prompt

def select_action(object_class, action_set, positive_action):
    base_message = f"To train a model that distinguishes between positive and negative images, you need to choose {NUM_POSITIVE} negative actions from the following negative action list. When choosing negative actions, you should consider the available actions from the object. For example, if the object is 'bird', possible actions are 'chase', 'feed', 'no_interaction', 'watch', etc. If the object is 'orange', possible actions are 'cut', 'hold', 'no_interaction', 'peel', etc. Therefore, you should choose hard negative actions that are clearly distinguishable from positive actions among the possible actions. \n- object: {object_class}\n- positive action: {positive_action}\n- negative action list: {action_set}\n\nPlease select a total of {NUM_POSITIVE} negative actions. The response format must be strictly result: ['negative_action1', 'negative_action2', ...], and all negative actions must be included in the negative action list.\n "
    
    # Generate 7 negative actions
    while True:
        try:
            response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "user", "content": base_message},
                ]
            )
            response_content = response.choices[0].message.content
            match = re.search(r"\[.*\]", response_content)
            result_list = ast.literal_eval(match.group())
        except Exception as e:
            print(e)
            print(f"Wrong response content: {response_content}")
            continue
        
        if len(result_list) == NUM_POSITIVE and all([action in action_set for action in result_list]):
            break
        else:
            print(f"Invalid response: {result_list}, retrying...")
    
    return result_list

json_split_path = '../../collections/Bongard-HOI/ma_splits/Bongard-HOI_train_seed1.json'
prompt_json_path = './prompts/hoi_diversified_new2.json'
temp_json_path = './prompts/hoi_diversified_new2_temp.json'
client = OpenAI()


if os.path.exists(temp_json_path):
    with open(temp_json_path, 'r') as f:
        results = json.load(f)
else:
    results = []

# Load dataset
with open(json_split_path, 'r') as f:
    dataset_list = json.load(f)
    
processed_ids = [result['id'] for result in results]

for dataset in tqdm(dataset_list):
    id = dataset['id']
    
    if id in processed_ids:
        print(f"Skipping id {id}")
        continue
    
    object_name = dataset['object_class'][0]; action_class_pos = dataset['action_class'][0]
    
    # Generate positive prompt
    positive_prompt = generate_prompt_from_pair(object_name, action_class_pos)
    
    # Generate negative prompts
    actions = ['adjust', 'blow', 'board', 'buy', 'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut_with', 'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'eat_with', 'exit', 'feed', 'fill', 'flush', 'fly', 'hit', 'hold', 'hop_on', 'hug', 'hunt', 'inspect', 'jump', 'kick', 'kiss', 'lasso', 'lick', 'lick_or_sip', 'lie_on', 'load', 'make', 'move', 'no_interaction', 'open', 'pack', 'park', 'peel', 'peel_or_cut', 'pet', 'repair', 'ride', 'run', 'sail', 'set', 'sign', 'sip', 'sit_at', 'sit_on', 'sit_under', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stir', 'straddle', 'straddle,pet', 'teach', 'tie', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear']
    actions.remove(action_class_pos)
    negative_actions = select_action(object_name, actions, action_class_pos)
    negative_prompts = []
    print(f"Positive prompt: {positive_prompt}")
    print(f"Negative actions: {negative_actions}")
    for action in tqdm(negative_actions, desc="Generating negative prompts"):
        prompt = generate_prompt_from_pair(object_name, action)
        negative_prompts.append(prompt)
    
    data_dict = {
        'id': id,
        'object_class': [object_name] * (NUM_POSITIVE * 2),
        'action_class': [action_class_pos] * NUM_POSITIVE + negative_actions,
        'positive_prompts': [positive_prompt] * NUM_POSITIVE,
        'negative_prompts': negative_prompts
    }
    
    results.append(data_dict)
    with open(temp_json_path, 'w') as f:
        json.dump(results, f)
    
with open(prompt_json_path, 'w') as f:
    json.dump(results, f)
