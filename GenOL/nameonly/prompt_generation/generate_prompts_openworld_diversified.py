import re
import json
from openai import OpenAI
from tqdm import tqdm

# Prompt format에 맞도록 수정해야 함 (0516)

def generate_prompt_positive(client, positive_prompt, previous_prompt_list):
    base_message = f"To create an image using a text-to-image generation model, I want to create a prompt. Below, a primary prompt for a positive image will be provided, and the goal is to generate one additional prompt that is nearly identical in meaning but have a slight variation in expression. For example, if the primary prompt is 'Dogs are running,' then similar prompt could be 'Dogs are sprinting,' 'Dogs are dashing,' 'Dogs are moving quickly,' etc. Here is a list of prompts that I have previously generated. Please create a new prompt that does not overlap with these."
    end_message = f"\nPositive prompt is '{positive_prompt}'. Please create one prompt sentence (under 10 words) that fits this description. Please ensure the response format is strictly 'prompt: answer'.\n"
    
    # For CoT Prompting
    for prompt in previous_prompt_list:
        base_message += f"\nprompt: {prompt}"
    base_message += end_message
    
    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": base_message},
            # {"role": "assistant", "content": "prompt: A photo of [concept] in neutral tones."},
            {"role": "user", "content": base_message},
        ]
    )
    response_content = response.choices[0].message.content
    match = re.search(r'prompt:\s*(.*)', response_content, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return response_content

def generate_prompt_negative(client, positive_prompt):
    base_message = f"To create an image using a text-to-image generation model, I want to create a prompt. Below, a prompt for a positive image will be provided, and the goal is to generate a prompt for a negative image. It is important that the negative prompt partially overlaps with the positive prompt and has slight differences. For example, if the positive prompt is 'Dogs are running', then 'Dogs are drinking water' would be the negative prompt.\n"
    end_message = f"\nPlease create 7 'negative' prompt sentences (under 5 words) that fits this description. Please ensure the response format is strictly 'prompt: answer'.\n Positive prompt: {positive_prompt}.\n"
    base_message += end_message

    # Generate one prompt using GPT-4 API
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "To create an image using a text-to-image generation model, I want to create a prompt. Below, a prompt for a positive image will be provided, and the goal is to generate a prompt for a negative image. It is important that the negative prompt partially overlaps with the positive prompt and has slight differences. For example, if the positive prompt is 'Dogs are running', then 'Dogs are drinking water' would be the negative prompt.\nPlease create one 'negative' prompt sentence (under 5 words) that fits this description. Please ensure the response format is strictly 'prompt: answer'.\n Positive prompt: A duck toy\n"},
            {"role": "assistant", "content": "prompt: A monster toy\nprompt: A robot toy\nprompt: Race car toys\nprompt: A dinosaur toy\nprompt: A teddy bear toy\nprompt: A unicorn toy\nprompt: A mermaid toy"}, # Example of 7 prompts
            {"role": "user", "content": base_message},
        ]
    )
    response_content = response.choices[0].message.content
    prompts = re.findall(r'prompt:\s*(.+)', response_content)
    return prompts


prompt_json_path = './prompts/openworld_base.json' # First stage result
NUM_NEG_PROMPTS = 7

jsonl_list = []
with open('../generate_twostage/train.jsonl', 'r') as f:
    for line in f:
        json_object = json.loads(line)
        jsonl_list.append(json_object)
        
client = OpenAI()
results_dict = {}

for data in tqdm(jsonl_list):
    data_dict = {}
    uid = data['uid']
    positive_primary_prompt = data['caption']
    
    # Generate positive prompts (using diversifying)
    positive_prompts = []
    for i in tqdm(range(6)):
        try:
            prompt = generate_prompt_positive(client, positive_primary_prompt, positive_prompts)
            print(f"Previous prompt list: {positive_prompts}")
            print(f"Generated positive prompt: {prompt}")
            positive_prompts.append(prompt)
        except Exception as e:
            breakpoint()
            pass
    
    positive_prompts.insert(0, positive_primary_prompt)
    # Generate negative prompts
    while True:
        negative_prompts = generate_prompt_negative(client, positive_primary_prompt)
        if len(negative_prompts) == 7:
            break
        else:
            print(f"Generate number of prompts is not 7! - {negative_prompts}")
    
    data_dict['positive_prompts'] = positive_prompts
    data_dict['negative_prompts'] = negative_prompts
    
    print(f"uid: {uid}, pos: {positive_prompts}"); print(f"uid: {uid}, neg: {negative_prompts}")
    results_dict[uid] = data_dict

with open(prompt_json_path, 'w') as f:
    json.dump(results_dict, f)
