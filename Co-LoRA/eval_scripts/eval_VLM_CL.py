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
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num
from tqdm import tqdm

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava

import warnings
import time
import datetime
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import StoppingCriteria, StoppingCriteriaList

ALPHABET = ['A','B','C','D','E','F']

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, repeat_len = 2):
      self.n = repeat_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        should_stop =False
        if input_ids.shape[1] > self.n*3:
            last_n_ids = input_ids[0][-self.n:]		# 마지막으로 생성한 n개의 토큰
            lastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            lastlastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            for i in range(self.n):
                if lastlastlast_n_ids[i] != lastlast_n_ids[i] or lastlast_n_ids[i] != last_n_ids[i]: # stop sequence와 비교
                    should_stop = False
                    break
                else :
                    should_stop = True
        return should_stop

def evaluate(dataset, dataname, round, model, tokenizer, device, max_new_tokens, training_args, logger, client_id=None, batch_size=2):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    
    # if 'llava' in model_args.model_name_or_path.lower():
    conv = conversation_lib_llava.default_conversation
    # elif 'bunny' in model_args.model_name_or_path.lower():
    #     conv = conversation_lib_bunny.default_conversation
    repeat_criteria = CustomStoppingCriteria()
    stop_str = conv.sep#2
    keywords = [stop_str]
    
    # img_feat_size = 729
    model.eval()
    predictions = []
    n_word_total = 0
    n_generated_word_total = 1
    n_word_correct = 1
    cnt = 0
    with torch.no_grad():
        for i, batch in enumerate((dataloader)): #tqdm
            inputs, golds, prompts = batch['input_ids'],  batch['gold'], batch['prompt']
            attention_mask = batch['attention_mask'].to(device=device)
            
            inputs = inputs.to(device=device, non_blocking=True)
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            
            input_dict = {
                'input_ids':inputs,
                'attention_mask':attention_mask,
                'do_sample': True if training_args.eval_temp > 0 else False,
                'temperature':training_args.eval_temp,
                'top_p':None,
                'num_beams':1,
                # 'top_p':0.75,
                # 'top_k':40,
                # 'num_beams':4,
                'max_new_tokens':max_new_tokens,
                'use_cache':True,
                'pad_token_id':tokenizer.eos_token_id,
                'stopping_criteria':stopping_criteria
            }
            
            if 'pixel_values' in batch.keys():
                imgs = batch['pixel_values']
                img_files = batch['image_file']
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                input_dict['pixel_values'] = imgs
            with torch.inference_mode():
                output_ids = model.generate(
                    **input_dict
                )
            # if 'vl' in model_args.model_name_or_path.lower():
            input_token_len = inputs.shape[1]
            output_ids = output_ids[:,input_token_len:]
            
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
            for batch_idx, (pred_sentence, gold, prompt) in enumerate(zip(pred_sentences, golds, prompts)):
                pred_sentence = pred_sentence.strip()
                input_label = tokenizer.encode(gold)
                output_id = tokenizer.encode(pred_sentence)
                n_word = len(set(input_label))
                n_generated_word = len(set(output_id))
                n_correct = matching_token_num(output_id, input_label)
                
                pred_dict = {"input":prompt, "sentence":pred_sentence, "gt_sentence":gold.strip()}
                if 'pixel_values' in batch.keys():
                    pred_dict['image_file'] = img_files[batch_idx]
                predictions.append(pred_dict)
                
                n_word_total += n_word
                n_generated_word_total += n_generated_word
                n_word_correct += n_correct
                cnt += 1

    scores = NLPEvaluator(predictions).evaluate()
    scores["precision"] = n_word_correct / n_word_total
    scores["recall"] = n_word_correct / n_generated_word_total
    
    predictions.append(scores)
    #save predictions
    if client_id is not None:
        logger.info(f"Test (Client id {client_id}) | Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
        if training_args.eval_iter is not None:
            with open(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{round}_iter{training_args.eval_iter}_{dataname}.json", 'w', encoding='utf-8') as fp:
                json.dump(predictions, fp, indent=4, ensure_ascii=False)
        else:
            with open(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{round}_{dataname}.json", 'w', encoding='utf-8') as fp:
                json.dump(predictions, fp, indent=4, ensure_ascii=False)
    else:
        logger.info(f"Test (Server) | Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
        with open(f"./eval_results/{training_args.mode}/{training_args.note}/server_round{round}_{dataname}.json", 'w', encoding='utf-8') as fp:
            json.dump(predictions, fp, indent=4, ensure_ascii=False)
    torch.cuda.empty_cache()
    
    return scores

def evaluate_choices(dataset, dataname, round, model, tokenizer, device, max_new_tokens, training_args, logger, client_id=None, batch_size=2):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))

    conv = conversation_lib_llava.default_conversation
    repeat_criteria = CustomStoppingCriteria()
    stop_str = conv.sep#2
    keywords = [stop_str]
    
    # img_feat_size = 729
    model.eval()
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate((dataloader)): #tqdm
            inputs, golds, prompts = batch['input_ids'], batch['gold'], batch['prompt']
            attention_mask = batch['attention_mask'].to(device=device)
            inputs = inputs.to(device=device, non_blocking=True)
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            
            input_dict = {
                'input_ids':inputs,
                'attention_mask':attention_mask,
                'do_sample': True if training_args.eval_temp > 0 else False,
                'temperature':training_args.eval_temp,
                'top_p':None,
                'num_beams':1,
                'max_new_tokens':max_new_tokens,
                'use_cache':True,
                'pad_token_id':tokenizer.eos_token_id,
                'stopping_criteria':stopping_criteria
            }
            
            if 'pixel_values' in batch.keys():
                imgs = batch['pixel_values']
                img_files = batch['image_file']
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                input_dict['pixel_values'] = imgs
            
            
            with torch.inference_mode():
                output_ids = model.generate(
                    **input_dict
                )
            # if 'vl' in model_args.model_name_or_path.lower():
            input_token_len = inputs.shape[1]
            output_ids = output_ids[:,input_token_len:]
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
            for batch_idx, (pred_sentence, gold, prompt) in enumerate(zip(pred_sentences, golds, prompts)):
                pred_sentence = pred_sentence.strip()
                
                choices = parse_choice_list(prompt)
                choices = [choice.strip() for choice in choices]
                if len(choices) > 0 and "'" in choices[0] and "'" not in gold:
                    choices = [choice.strip("'") for choice in choices]
                pred_option = can_infer(pred_sentence, choices)
                if isinstance(pred_option, str):
                    # if gold == pred_option:
                    if gold.lower() == pred_option.lower():
                        correct += 1
                        status='correct'
                    else:
                        status='wrong'
                else:
                    status = 'unkown'
                total += 1
                
                pred_dict = {"input":prompt, "sentence":pred_sentence, "gt_sentence":gold.strip(), 'status':status}
                if 'pixel_values' in batch.keys():
                    pred_dict['image_file'] = img_files[batch_idx]
                predictions.append(pred_dict)

    scores = {'accuracy': correct/total}
    
    predictions.append(scores)
    #save predictions
    if client_id is not None:
        logger.info(f"Test (Client id {client_id}) | Data {dataname} | accuracy {scores['accuracy']} |")
        if training_args.eval_iter is not None:
            with open(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{round}_iter{training_args.eval_iter}_{dataname}.json", 'w', encoding='utf-8') as fp:
                json.dump(predictions, fp, indent=4, ensure_ascii=False)
        else:
            with open(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{round}_{dataname}.json", 'w', encoding='utf-8') as fp:
                json.dump(predictions, fp, indent=4, ensure_ascii=False)
    else:
        logger.info(f"Test (Server) | Data {dataname} | accuracy {scores['accuracy']} |")
        with open(f"./eval_results/{training_args.mode}/{training_args.note}/server_round{round}_{dataname}.json", 'w', encoding='utf-8') as fp:
            json.dump(predictions, fp, indent=4, ensure_ascii=False)
    torch.cuda.empty_cache()
    
    return scores

def parse_choice_list(input_string):
    # Try to find the choice list in the format "Choice list:[...]"
    match = re.search(r'Choice list:\[(.*?)\]', input_string)
    if match:
        # comics_dialogue & textcloze
        choices = [choice.strip() for choice in match.group(1).split('|')]
        if len(choices) > 2:
            return ALPHABET[:len(choices)]
        
        # Split the choices and strip whitespace
        choices = [choice.strip() for choice in match.group(1).split(',')]
        # If choices start with "Image", only keep the "Image X" part
        if all(choice.startswith("Image A") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        elif all(choice.startswith("Image 1") for choice in choices):
            choices = [re.match(r'(Image [1-8])', choice).group(1) for choice in choices]
        return choices

    match = re.search(r'Choice lists: \[(.*?)\]', input_string)
    if match:
        # comics_dialogue & textcloze
        choices = [choice.strip() for choice in match.group(1).split('|')]
        if len(choices) > 2:
            return ALPHABET[:len(choices)]
        
        # Split the choices and strip whitespace
        choices = [choice.strip() for choice in match.group(1).split(',')]
        # If choices start with "Image", only keep the "Image X" part
        if all(choice.startswith("Image A") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        elif all(choice.startswith("Image 1") for choice in choices):
            choices = [re.match(r'(Image [1-8])', choice).group(1) for choice in choices]
        return choices

    match = re.search(r'Choice list: \[(.*?)\]', input_string)
    if match:
        # comics_dialogue & textcloze
        choices = [choice.strip() for choice in match.group(1).split('|')]
        if len(choices) > 2:
            return ALPHABET[:len(choices)]
        
        # Split the choices and strip whitespace
        choices = [choice.strip() for choice in match.group(1).split(',')]
        # If choices start with "Image", only keep the "Image X" part
        if all(choice.startswith("Image A") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        elif all(choice.startswith("Image 1") for choice in choices):
            choices = [re.match(r'(Image [1-8])', choice).group(1) for choice in choices]
        return choices
    
    match = re.search(r'Choice List: \[(.*?)\]', input_string)
    if match:
        # Split the choices and strip whitespace
        choices = [choice.strip() for choice in match.group(1).split(',')]
        # If choices start with "Image", only keep the "Image X" part
        if all(choice.startswith("Image ") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        return choices

    match = re.search(r'\n \[(.*?)\]\n', input_string)
    if match:
        # Split the choices and strip whitespace
        choices = [choice.strip() for choice in match.group(1).split(',')]
        # If choices start with "Image", only keep the "Image X" part
        if all(choice.startswith("Image ") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        return choices
    
    # If not found, try to find choices in the format "A. ... B. ... C. ... D. ..."
    match = re.findall(r'([A-D])\.\s*(.*?)(?=\n[A-D]\.|$)', input_string, re.DOTALL)
    if match:
        return [letter for letter, _ in match]
    
    # If still not found, look for "Image A: ..., Image B: ..., Image C: ..., Image D: ..."
    match = re.findall(r'Image ([A-D]):', input_string)
    if match:
        return [f"Image {letter}" for letter in match]
    
    # wcvqa
    if 'Print only the answer with a single answer id' in input_string:
        return ['1','2','3','4','5']
    
    # If no choices found, return an empty list
    return []

def can_infer(answer, choices):
    if len(answer) == 0:
        return False
    answer = str(answer).lower()
    
    # Special case for ['Positive', 'Negative']
    if set(choices) == {'Positive', 'Negative'}:
        if 'yes' in answer or 'Yes' in answer:
            return 'Positive'
        elif 'no' in answer or 'No' in answer:
            return 'Negative'
    
    # First, look for exact matches if choices are not simple letters
    if not all(len(choice) == 1 and choice in string.ascii_uppercase for choice in choices):
        for choice in choices:
            if choice.lower() in answer or choice in answer:  # Allow for case-insensitive exact match
                return choice
    
    # Then, look for simple letter matches (A, B, C, ...)
    letter_matches = re.findall(r'\b[A-Z]\b', answer.upper())
    for letter in letter_matches:
        index = string.ascii_uppercase.index(letter)
        if index < len(choices):
            return choices[index]
    
    # If choices are simple letters, look for those
    if all(len(choice) == 1 and choice in string.ascii_uppercase for choice in choices):
        for choice in choices:
            if choice in answer.upper():
                return choice
            
    # remove underscore and try
    answer =  answer.strip().replace('_', ' ').lower()
    normalized_choices = [choice.replace('_', ' ').lower() for choice in choices]
    if answer in normalized_choices:
        return choices[normalized_choices.index(answer)]
    # Check for partial matches
    for i, choice in enumerate(normalized_choices):
        if answer in choice or choice in answer:
            return choices[i]
    
    
    # If no match found, return False
    return False

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

    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}/federated')

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
    print(device)
    logger.info(f"Set the device ({device})")

    # model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    _, test_datalists = get_datalists(training_args, training_args.scenario)
    
    batch_size = 4 #if 'l2p' in training_args.mode or 'dap' in training_args.mode or 'LAE' in training_args.mode or 'fedsim' in training_args.mode else 4
    
    logger.info(f'Evaluatiing clients at round {training_args.round_to_eval}')
    start_time = time.time()

    if training_args.unseen_task:
        # if you want, load the model weight trained on seen tasks to evaluate on unseen tasks, you can load the model weight here. Otherwise, it will evaluate the base model (zeroshot)
        test_datalist = test_datalists[0] # assume only one client
        for data_info in test_datalist:
            print(data_info['data_name'])
            dataset = GenerationDataset(data_info['data'], tokenizer, data_args, processor)
            if data_info['type'] == 'open-ended':
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, None, batch_size)
            elif data_info['type'] == 'multi-choice':
                evaluate_choices(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, None, batch_size)
            else:
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, None, batch_size)
        return
    
    for client_id in range(training_args.num_clients):
        if training_args.eval_client is not None:
            if client_id != training_args.eval_client:
                continue
        if training_args.eval_client_start is not None and training_args.eval_client_end is not None:
            if client_id < training_args.eval_client_start or client_id > training_args.eval_client_end:
                continue
        # load client weight
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
        
        test_datalist = test_datalists[client_id]
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = test_datalist[0]['model_id']
        new_data_args = copy.deepcopy(data_args)
        new_data_args.model_name_for_dataarg = test_datalist[0]['model_id']
        model, tokenizer, processor, data_args = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, new_data_args)
        print(f"Model name: {test_datalist[0]['model_id']}")
        if training_args.eval_all:
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
            
            for client_id_ in range(training_args.num_clients):
                if training_args.eval_client_eval_start is not None and training_args.eval_client_eval_end is not None:
                    if client_id_ < training_args.eval_client_eval_start or client_id_ > training_args.eval_client_eval_end:
                        continue
                test_datalist_ = test_datalists[client_id_]
                for data_info in test_datalist_:
                    dataset = GenerationDataset(data_info['data'], tokenizer, data_args, processor)
                    if data_info['type'] == 'dummy':
                        if training_args.eval_iter is not None:
                            output_path = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_iter{training_args.eval_iter}_{data_info['data_name']}.json"
                        else:
                            output_path = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_{data_info['data_name']}.json"
                        with open(output_path, 'w') as fp:
                            json.dump([{'accuracy':-1}], fp)
                        continue
                    if (training_args.eval_iter is not None and os.path.isfile(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_iter{training_args.eval_iter}_{data_info['data_name']}.json")) \
                        or (training_args.eval_iter is None and os.path.isfile(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_{data_info['data_name']}.json")):
                        print('output file already exist')
                        continue
                    
                    if data_info['type'] == 'open-ended':
                        evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, client_id, batch_size)
                    elif data_info['type'] == 'multi-choice':
                        evaluate_choices(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, client_id, batch_size)
                    else:
                        evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, client_id, batch_size)
            continue
        
        for data_info in test_datalist:
            
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
            
            dataset = GenerationDataset(data_info['data'], tokenizer, data_args, processor)
            
            if data_info['type'] == 'dummy':
                if training_args.eval_iter is not None:
                    output_path = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_iter{training_args.eval_iter}_{data_info['data_name']}.json"
                else:
                    output_path = f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_{data_info['data_name']}.json"
                with open(output_path, 'w') as fp:
                    json.dump([{'accuracy':-1}], fp)
                continue
            # if training_args.mode not in ['fedsim', 'feddat']:
            if (training_args.eval_iter is not None and os.path.isfile(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_iter{training_args.eval_iter}_{data_info['data_name']}.json")) \
                or (training_args.eval_iter is None and os.path.isfile(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{training_args.round_to_eval}_{data_info['data_name']}.json")):
                print('output file already exist')
                continue
                
            if data_info['type'] == 'open-ended':
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, client_id, batch_size)
            elif data_info['type'] == 'multi-choice':
                evaluate_choices(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, client_id, batch_size)
            else:
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args.max_new_tokens, training_args, logger, client_id, batch_size)
            
        del model
    
    logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
def get_datalists(args, scenario_num):
    with open(f"./scenarios/{scenario_num}.json") as fp:
        scenario = json.load(fp)
    incremental_setup = {
        "num_active_clients": [args.num_clients,]*int(args.num_rounds * args.num_tasks)
    }
    assert args.num_clients == len(scenario)

    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        eval_cnt = 0
        train_cnt = 0
        for data in client_data['datasets']:
            if data['dataset'] == 'dummy':
                for i in range(rounds_per_task):
                    train_datalist.append({'datalist':[],'model_id':''})
                test_datalist.append({'data_name':'dummy','data':[],'type':'dummy','model_id':client_data['model_id']})
                continue
            
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            for i in range(int(rounds_per_task)):
                train_datalist.append(
                    {'datalist':datalist[i*samplenum_per_rounds:(i+1)*samplenum_per_rounds],
                     'train_cnt': train_cnt + samplenum_per_rounds})
                train_cnt += samplenum_per_rounds
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "type": data['type'] if 'type' in data else 'open-ended',
                "model_id": client_data['model_id'],
                "data": datalist,# if scenario_num not in [90,91,92,93] else datalist[:15],
                "eval_cnt": eval_cnt})
            eval_cnt += len(datalist)
            
            train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist

    return train_datalists, test_datalists

def anytime_evaluation(model, tokenizer, processor, test_datalist, eval_batchsize, curr_round, client_id, training_args, max_new_tokens, data_args, logger):
    torch.cuda.empty_cache()
    final_score = 0
    task_count = 0
    for data_info in test_datalist:
        # if train_datalists[client_id][training_args.round_to_eval-1]['train_cnt'] > data_info['eval_cnt']:
        if curr_round >= data_info['train_start_round']:
            print(f"eval {data_info['data_name']}")
            dataset = GenerationDataset(data_info['data'], tokenizer, data_args, processor)
            if data_info['type'] == 'open-ended':
                result = evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, model.device, max_new_tokens, training_args, logger, client_id, eval_batchsize)
            elif data_info['type'] == 'multi-choice':
                result = evaluate_choices(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, model.device, max_new_tokens, training_args, logger, client_id, eval_batchsize)
            else:
                result = evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, model.device, max_new_tokens, training_args, logger, client_id, eval_batchsize)
            
            if data_info['type'] == 'open-ended':
                if data_info['metric'] == 'F1':
                    score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                elif data_info['metric'] == 'RougeL':
                    score = result['ROUGE_L'][0]
            elif data_info['type'] == 'multi-choice':
                score = result['accuracy']
            
            final_score += score
            task_count += 1
    torch.cuda.empty_cache()
    return final_score / task_count
        
if __name__ == "__main__":
    main()
