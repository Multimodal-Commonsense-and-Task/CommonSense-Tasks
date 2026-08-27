import logging.config
import os
import random
import re
import string

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingConfig
import transformers
from utils.train_utils import get_VLMmodel
import copy

import json
import gzip
import pandas as pd
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num
from tqdm import tqdm

from models.llava import conversation as conversation_lib_llava
from eval_VLM_CL import get_datalists

import warnings
import time
import datetime
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import tarfile
from mmlu_categories import categories, subcategories

transformers.logging.set_verbosity(40)

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    ll = subject.split("_")
    s = ""
    for entry in ll:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    # prompt += "
    if include_answer:
        prompt += "\nAnswer: {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
import ssl
import urllib.request
import os.path as osp
def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path



@torch.no_grad()
def eval(subject, model, processor, tokenizer, dev_df, test_df, device):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = 5
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        conversation = [{'role':'user','content':prompt}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        prompt += "Answer: "
        input_ids = processor(text=prompt, return_tensors='pt')['input_ids'].to(device)
        while input_ids.shape[-1] > 1024:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            conversation = [{'role':'user','content':prompt}]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            prompt += "Answer: "
            input_ids = processor(text=prompt, return_tensors='pt')['input_ids'].to(device)

        label = test_df.iloc[i, test_df.shape[1] - 1]
        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("A").input_ids[-1]],
                logits[tokenizer("B").input_ids[-1]],
                logits[tokenizer("C").input_ids[-1]],
                logits[tokenizer("D").input_ids[-1]],
            ]).float(),
            dim=0,
        ).detach().cpu().numpy())
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

@torch.no_grad()
def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()
    
    os.makedirs(f"eval_results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'eval_results/{training_args.mode}/{training_args.note}/round_{training_args.round_to_eval}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")
    
    _, test_datalists = get_datalists(training_args, training_args.scenario)
    for client_id in range(training_args.num_clients):
        if training_args.eval_client:
            if client_id != training_args.eval_client:
                continue
        if not training_args.zeroshot:
            try:
                if training_args.eval_iter is not None:
                    logger.info(f'load ./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}_itr{training_args.eval_iter}.pth')
                    client_state_dict = torch.load(f'./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}_itr{training_args.eval_iter}.pth', map_location='cpu')    
                else:
                    logger.info(f'load ./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}.pth')
                    client_state_dict = torch.load(f'./client_states_{training_args.note}/{client_id}_client_model_round{training_args.round_to_eval}.pth', map_location='cpu')
            except Exception as e:
                print(e)
                continue
        results_file = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_mmlu.json"
        if os.path.isfile(results_file):
            print('output file already exist')
            continue
        
        test_datalist = test_datalists[client_id]
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = test_datalist[0]['model_id']
        new_data_args = copy.deepcopy(data_args)
        new_data_args.model_name_for_dataarg = test_datalist[0]['model_id']
        model, tokenizer, processor, data_args = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, new_data_args)
        conv = conversation_lib_llava.default_conversation.copy()
        if conv.version == 'qwen':
            processor.chat_template = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""
        elif conv.version == 'llama3':
            processor.chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{%- if system_message %}\n    {{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n    {%- if tools is not none %}\n        {{- "Environment: ipython\\n" }}\n    {%- endif %}\n    {{- "Cutting Knowledge Date: December 2023\\n" }}\n    {{- "Today Date: " + date_string + "\\n\\n" }}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n        {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n        {{- "Do not use variables.\\n\\n" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- "\\n\\n" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- "<|eot_id|>" }}\n{%- endif %}\n\n{%- if tools_in_user_message and not tools is none %}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}\n    {%- endif %}\n    {{- "<|start_header_id|>user<|end_header_id|>\\n\\n" }}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>" }}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- "<|start_header_id|>" + message['role'] + "<|end_header_id|>\\n\\n" + message['content'] | trim + "<|eot_id|>" }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n        {{- '{"name": "' + tool_call.name + '", ' }}\n        {{- '"parameters": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n    {{- "<|start_header_id|>assistant<|end_header_id|>\\n\\n" }}\n{%- endif %}\n"""

        if not training_args.zeroshot:
            model.load_state_dict(client_state_dict, strict=False)
        model = model.to(torch.bfloat16)
        model = model.to(device)
        
        if 'perada' in training_args.mode or 'ditto' in training_args.mode:
            model.set_state('lora2')
        elif 'feddpa' in training_args.mode or 'feddat' in training_args.mode:
            model.set_state('gate')
        elif 'fedours' in training_args.mode or 'dual' in training_args.mode or 'fedquad' in training_args.mode or 'fedhexa' in training_args.mode or 'feddat' in training_args.mode:
            model.set_state(training_args.set_state)
        
        # Get test file
        if not os.path.exists("dataset/mmlu/data"):
            os.makedirs("dataset/mmlu/", exist_ok=True)
            download_url("https://people.eecs.berkeley.edu/~hendrycks/data.tar",
                        'dataset/mmlu')
            t = tarfile.open("dataset/mmlu/data.tar", "r:")
            t.extractall(path="dataset/mmlu/")
            t.close()

        data_dir = os.path.join('dataset', "mmlu/data")
        
        subjects = sorted([
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f
        ])
        
        output_dir = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_mmlu"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        all_cors = []
        subcat_cors = {
            subcat: []
            for subcat_lists in subcategories.values() for subcat in subcat_lists
        }
        cat_cors = {cat: [] for cat in categories}
        
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:5]
            test_df = pd.read_csv(os.path.join(data_dir, "test",
                                            subject + "_test.csv"),
                                header=None)

            cors, acc, probs = eval(subject, model, processor, tokenizer, dev_df, test_df,
                                device)
            subcats = subcategories[subject]
            for subcat in subcats:
                subcat_cors[subcat].append(cors)
                for key in categories.keys():
                    if subcat in categories[key]:
                        cat_cors[key].append(cors)
            all_cors.append(cors)
            test_df["{}_correct".format(output_dir)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(output_dir,
                                                choice)] = probs[:, j]
            test_df.to_csv(
                os.path.join(output_dir,"{}.csv".format(subject)),
                index=None,
            )
        
        results = {"subcategories": {}, "categories": {}}
        for subcat in subcat_cors:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

        for cat in cat_cors:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            results["categories"][cat] = cat_acc
            print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        weighted_acc = np.mean(np.concatenate(all_cors))
        results["weighted_accuracy"] = weighted_acc
        print("Average accuracy: {:.3f}".format(weighted_acc))
        
        with open(results_file, "w") as f:
            json.dump([results, {"accuracy":weighted_acc}], f)
        
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
if __name__ == "__main__":
    main()