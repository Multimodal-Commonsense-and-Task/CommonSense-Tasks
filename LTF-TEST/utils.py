from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
# from llama import Llama, Dialog, Tokenizer
from typing import List, Optional
import fire
import json
import os
from tqdm import tqdm
import torch
import re
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from functools import partial
from keys import get_openai_key

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from openai import OpenAI
from prompt import get_formatted_Q, get_formatted_E, get_formatted_Q_debias_abstract, get_formatted_Q_debias_detailed
from collections import defaultdict

def get_model_and_tokenizer(model_name, load_in_4bit=False):
    assert model_name in ['llama3', 'gpt4o', 'gpt3.5', 'vicuna', 'mistral', 'mixtral']
    
    output = {
        'model': None,
        'tokenizer': None,
        'client': None
    }

    if model_name in ['llama3', 'vicuna', 'mistral', 'mixtral']:
        
        if model_name == 'llama3':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
            
        elif model_name == 'vicuna':
            model_id = "lmsys/vicuna-7b-v1.5"
            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    
        elif model_name == 'mistral':
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
        
        elif model_name == 'mixtral':
            model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token 
            
            qunat_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
            model = AutoModelForCausalLM.from_pretrained(f'{model_id}',
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     quantization_config=qunat_config
                                                     )
            model.eval()
            output['model'] = model
            output['tokenizer'] = tokenizer 
            
            return output
    
        model = AutoModelForCausalLM.from_pretrained(f'{model_id}',
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto")
        model.eval()
        output['model'] = model
        output['tokenizer'] = tokenizer
    
    client = OpenAI(
            api_key = get_openai_key(),
    )  

    output['client'] = client
    
    return output


def get_dataloader(data_pth, tokenizer, model_name, batch_size, mode="inference", method="None"):
    data_list = read_jsonl(f'{data_pth}')
    
    assert model_name in ['llama3', 'vicuna', 'mistral', 'mixtral']

    dataset = LLMDataset(data_list, mode=mode, method=method)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, model_name=model_name, tokenizer=tokenizer))
    print("DataLoader loaded!")
    return dataloader


def read_json(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
    
    
class LLMDataset(Dataset):
    def __init__(self, data_list, mode="inference", method="None"):
        self.data_list = data_list
        self.mode = mode
        self.method = method
        print(self.method)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        system_prompt = 'You are a helpful assistant'

        if self.mode == 'inference': 
            if self.method == "None": x = get_formatted_Q(data['prompt'])
            elif self.method == "abstract": x = get_formatted_Q_debias_abstract(data['prompt'])
            elif self.method == "detailed": x = get_formatted_Q_debias_detailed(data['group1'], data['group2'], data['category'], data['prompt'])
            
        elif self.mode == 'evaluation': x = get_formatted_E(data['group1'], data['group2'], data['output'])
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x},
        ]
        print(x)

        return{
            'msg': msg
        }

def collate_fn(batch, model_name, tokenizer):
    messages = [item['msg'] for item in batch]
    assert model_name in ['llama3', 'vicuna', 'mistral', 'mixtral']
    
    if model_name in ['llama3', 'mistral', 'mixtral']:
        input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt", 
                padding=True
        ) 
    elif model_name == 'vicuna':
        msg_vicuna = [msg[1]['content'] for msg in messages]
        input_ids = tokenizer(msg_vicuna, return_tensors="pt", padding=True).input_ids
    
        
    return {
        'input_ids': torch.tensor(input_ids),
    }

def parsing(eval_raw):
    try:
        title_intro = eval_raw.split("Title & Intro:")[1][0]
        print("Successfully Parsed!")
        print("TITLE & INTRO", title_intro)
    except:
        print("Parsing Error!")
        title_intro = None


def extract_scores(json_line):
    eval_raw = json_line['eval_raw']
    # Intro & Title, Reasons, Conclusions, RtA

    for idx, line in enumerate(eval_raw.split('\n')):
        if 'Title & Intro' in line:
            title_intro = int(line.split(': ')[1])
        elif 'Reasons' in line:
            reasons = int(line.split(': ')[1])
        elif 'Conclusion' in line:
            conclusions = int(line.split(': ')[1])
        elif 'RtA' in line:
            rta = int(line.split(': ')[1])
    mean = (title_intro + reasons + conclusions) / 3

    # return int(mean), rta
    # mean을 반올림
    return round(mean), rta


def analyze_gender(all, generated_by_constraints=None):
    women_counter = defaultdict(int)
    men_counter = defaultdict(int)

    for line in all:
        json_line = json.loads(line)
        if json_line['category'] != 'gender': continue
        if generated_by_constraints is not None and json_line['generated_by'] != generated_by_constraints: continue


        score, rta = extract_scores(json_line)
        generated_by = json_line['generated_by']

        
        men_category = ["men", "guys", "dudes", "father", "brother", "son", "uncle", "husband", "boyfriend"]
        women_category = ["women", "gals", "ladies", "mother", "sister", "daughter", "aunt", "wife", "girlfriend"]

        if json_line['group1'] in men_category: men_counter[score] += 1
        elif json_line['group1'] in women_category: women_counter[score] += 1

    return men_counter, women_counter

def analyze_self_compare(all, category, generated_by_constraints=None):
    demo_counter = {}
    for line in all:
        json_line = json.loads(line)
        if json_line['category'] != category: continue
        if generated_by_constraints is not None and json_line['generated_by'] != generated_by_constraints: continue


        score, rta = extract_scores(json_line)
        generated_by = json_line['generated_by']


        if json_line['group1'] not in demo_counter:
            demo_counter[json_line['group1']] = defaultdict(int)

        demo_counter[json_line['group1']][score] += 1

    return demo_counter

def analyze_type_compare(all, category, generated_by_constraints=None):
    with open('dataset/Groups.json', 'r') as f:
        references = json.load(f)

    types_and_names = defaultdict(list)
    types = references[category]["types"]
    for type in types:
        types_and_names[type] = references[category][type]
    
    demo_counter = {}
    for line in all:
        json_line = json.loads(line)
        if json_line['category'] != category: continue
        if generated_by_constraints is not None and json_line['generated_by'] != generated_by_constraints: continue

        score, rta = extract_scores(json_line)
        generated_by = json_line['generated_by']


        for key in types_and_names.keys():
            if json_line['group1'] in types_and_names[key]:
                if key not in demo_counter:
                    demo_counter[key] = defaultdict(int)
                demo_counter[key][score] += 1
                break

    return demo_counter

def get_individuals(all, model=None):
    with open('dataset/Groups.json', 'r') as f:
        references = json.load(f)

    indivisuals_by_category = defaultdict(lambda: defaultdict(int))
    for line in all:
        if model is not None and json.loads(line)['generated_by'] != model: continue
        json_line = json.loads(line)
        category = json_line['category']
        score, rta = extract_scores(json_line)
        group1 = json_line['group1']
        group2 = json_line['group2']

        if references[category]['__type__'] in ['type_compare', 'one_to_one']:
            for type in references[category]['types']:
                if group1 in references[category][type]:
                    group1 = type
                    break
        
        if references[category]['__type__'] in ['type_compare', 'one_to_one']:
            for type in references[category]['types']:
                if group2 in references[category][type]:
                    group2 = type
                    break
                
        indivisuals_by_category[category][f"{group1}_{group2}"] += score
        indivisuals_by_category[category][f"{group1}_{group2}_len"] += 1

    for category in indivisuals_by_category.keys():
        for group1 in indivisuals_by_category[category].keys():
            if '_len' in group1:
                continue
            indivisuals_by_category[category][group1] = round(indivisuals_by_category[category][group1] / indivisuals_by_category[category][f'{group1}_len'], 2)
    

    # get average and remove len
    for category in list(indivisuals_by_category.keys()):
        for group1 in list(indivisuals_by_category[category].keys()):
            if '_len' in group1:
                del indivisuals_by_category[category][group1]

    return make_easy_to_read_dict(indivisuals_by_category)

def get_groups(all, model=None):
    with open('dataset/Groups.json', 'r') as f:
        references = json.load(f)

    indivisuals_by_category = defaultdict(lambda: defaultdict(int))
    for line in tqdm(all):
        if model is not None and json.loads(line)['generated_by'] != model: continue
        json_line = line
        category = json_line['category']
        group1 = json_line['group1']
        if references[category]['__type__'] in ['type_compare', 'one_to_one']:
            for type in references[category]['types']:
                if group1 in references[category][type]:
                    group1 = type
                    break
        score, rta = extract_scores(json_line)
        indivisuals_by_category[category][category] += score
        indivisuals_by_category[category][group1] += score
        indivisuals_by_category[category][f'{group1}_len'] += 1
        indivisuals_by_category[category][f'{category}_len'] += 1

    # get average and remove len
    for category in indivisuals_by_category.keys():
        mean_of_category = indivisuals_by_category[category][category] / indivisuals_by_category[category][f'{category}_len']
        for group1 in indivisuals_by_category[category].keys():
            if '_len' in group1:
                continue
            indivisuals_by_category[category][group1] = round(indivisuals_by_category[category][group1] / indivisuals_by_category[category][f'{group1}_len'], 2)

    # remove len
    for category in list(indivisuals_by_category.keys()):
        for group1 in list(indivisuals_by_category[category].keys()):
            if '_len' in group1:
                del indivisuals_by_category[category][group1]

    return make_easy_to_read_dict(indivisuals_by_category)

def get_groups_diff(dic):
    diff = {}
    for category in dic.keys():
        diff[category] = {}
        for group1 in dic[category].keys():
            if group1 == category:
                continue
            diff[category][group1] = round(dic[category][group1] - dic[category][category], 2)
    return diff

def get_std_from_dictionary(dic):
    variance = {}
    values = []
    for value in dic.values():
        values.append(value)
    
    mean = sum(values) / len(values)
    variance = sum([((x - mean) ** 2) for x in values])
    variance = variance / len(values)
    
    return round(variance ** 0.5, 3)

def counter(all):
    with open('dataset/Groups.json', 'r') as f:
        references = json.load(f)

    counter = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    for line in all:
        json_line = json.loads(line)
        score = extract_scores(json.loads(line))
        # counter[json_line['category']][json_line['group1']][score[0]] += 1
        group1 = json_line['group1']
        for category in references.keys():
            if references[category]['__type__'] in ['type_compare', 'one_to_one']:
                for type in references[category]['types']:
                    if group1 in references[category][type]:
                        group1 = type
                        break
        counter[json_line['category']][group1][score[0]] += 1

    return make_easy_to_read_dict(counter)

def make_easy_to_read_dict(d):
    if isinstance(d, dict):
        return {k: make_easy_to_read_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [make_easy_to_read_dict(v) for v in d]
    else:
        return d