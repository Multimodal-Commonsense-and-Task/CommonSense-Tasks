import logging.config
import os
import random
import re
import string

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel

# from utils.method_manager_VLM import select_method
# from torch.utils.tensorboard import SummaryWriter

from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig

# from utils.data_loader_VLM import GenerationDataset, Qwen_GenerationDataset, DataCollatorForGenerationDataset, Qwen_DataCollatorForGenerationDataset
from utils.data_loader_VLM import LLM_GenerationDataset, LLM_DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num#, can_infer
from tqdm import tqdm

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava
from models.bunny import conversation as conversation_lib_bunny
from models.duallora.dualloralayer import DualLoraLayer
from models.dual_ia3.dual_ia3_layer import DualIA3Layer

import warnings
import time
import datetime
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import StoppingCriteria, StoppingCriteriaList

ALPHABET = ['A','B','C','D','E','F']
CLASSIFICAION = {"SuperNI": {4: ['positive', 'negative'], 11: ['joy', 'love', 'anger', 'fear', 'surprise'], 12: ['POS', 'NEG'], 13:['A', 'B', 'C', 'D']},
                 "longsequence": {1: ['contradiction', 'entailment'], 2: ['True', 'False'], 3: ['very negative', 'negative', 'neutral', 'positive', 'very positive'], 6:['contradiction', 'entailment', 'neutral'], 7:['Good', 'Bad'], 8:['True', 'False'], 9:[['contradiction', 'entailment', 'neutral']], 10:['very negative', 'negative', 'neutral', 'positive', 'very positive'], 11:["A", "B"], 12:["yes", "no"], 13: ["Good", "Bad"], 14:["True", "False"]},
                 "longsequence2": {1: ['contradiction', 'entailment'], 2: ['True', 'False'], 3: ['very negative', 'negative', 'neutral', 'positive', 'very positive'], 6:['contradiction', 'entailment', 'neutral'], 7:['Good', 'Bad'], 8:['True', 'False'], 9:[['contradiction', 'entailment', 'neutral']], 10:['very negative', 'negative', 'neutral', 'positive', 'very positive'], 11:["A", "B"], 12:["yes", "no"], 13: ["Good", "Bad"], 14:["True", "False"]},
                 "trace": {3: ['A', 'B', 'C'], 4: ['A', 'B', 'C']},
                 "citb": {0: ['0', '1'], 10: ["Cheap", "Moderate", "Expensive","Don't Care"], 15:["accepted", "rejected"], 17:["Yes", "No"], 18:["book", 'cancel', 'no_flight', 'no_reservation']},
                 "glue": {0: ["Negative", "Postive"], 1: ["society and culture", "science", "health", "education and reference",
                            "computers and internet", "sports", "business", "entertainment and Music",
                            "family and relationships", "politics and government"], 2: ["not_entailment", "not_entailment"], 3: ["entailment", "neutral", "contradiction"]},
                 "standard_cl": {0: ["world", "sports", "business", "science"], 1: ["No", "Yes"], 2: ["company", "educationalinstitution", "artist", "athlete", "officeholder",
                        "meanoftransportation", "building", "naturalplace", "village", "animal",
                        "plant", "album", "film", "writtenwork"], 3: ["terrible", "bad", "middle", "good", "wonderful"]}
}


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

def evaluate(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=1):
    
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    conv = conversation_lib_llava.default_conversation
    # if 'llama' in model_args.model_name_or_path.lower():
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
    # elif 'qwen' in model_args.model_name_or_path.lower():
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=LLM_DataCollatorForGenerationDataset(tokenizer))
    stop_str = conv.sep
    
    repeat_criteria = CustomStoppingCriteria()
    keywords = [stop_str]
    
    # img_feat_size = 729
    predictions = []
    n_word_total = 0
    n_generated_word_total = 1
    n_word_correct = 1
    cnt = 0
    
    for i, batch in enumerate((dataloader)): #tqdm
        # if "Qwen" in model_args.model_name_or_path:
        #     imgs = None
        #     if 'pixel_values' in batch.keys():
        #         imgs, img_files, image_grid_thw = batch['pixel_values'], batch['image_file'], batch['image_grid_thw']
        #     inputs, golds= batch['input_ids'], batch['gold']
        # else:
        #     if 'image_file' in batch.keys():
        #         imgs, img_files = batch['images'], batch['image_file']
        #     inputs, golds, prompts = batch['input_ids'], batch['gold'], batch['prompt']
        imgs = None
        if 'pixel_values' in batch.keys():
            imgs, img_files, image_grid_thw = batch['pixel_values'], batch['image_file'], batch['image_grid_thw']
        inputs, golds= batch['input_ids'], batch['gold']
        attention_mask = batch['attention_mask'].to(device=device)
        
        inputs = inputs.to(device=device, non_blocking=True)
        if imgs is not None:
            if isinstance(imgs, list):
                imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
            else:
                imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            image_sizes = [x.shape[-2:] for x in imgs]
        # keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
        # stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])

        # if "Qwen" in model_args.model_name_or_path:
        #     with torch.inference_mode():
        #         output_ids = model.generate(
        #             inputs[0],
        #             attention_mask=attention_mask[0],
        #             do_sample=True,# if args.temperature > 0 else False,
        #             temperature=training_args.eval_temp,#args.temperature,
        #             top_p=None,#args.top_p,
        #             num_beams=1,#args.num_beams,
        #             max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
        #             use_cache=True,
        #             pad_token_id=tokenizer.eos_token_id,
        #             stopping_criteria = stopping_criteria,
        #             # prompt=prompts if training_args.is_prompt else None,
        #         )
        # else:
        with torch.inference_mode():
            output_ids = model.generate(
                inputs,
                attention_mask=attention_mask,
                do_sample=True,# if args.temperature > 0 else False,
                temperature=training_args.eval_temp,#args.temperature,
                top_p=None,#args.top_p,
                num_beams=1,#args.num_beams,
                max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                # stopping_criteria = stopping_criteria,
            )
        # if 'bunny' in model_args.model_name_or_path.lower():
        #     input_token_len = inputs.shape[1]
        #     output_ids = output_ids[:,input_token_len:]
        input_token_len = inputs.shape[1]
        if input_token_len < len(output_ids[0]):
            output_ids = output_ids[:,input_token_len:]
        
        pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
        
        del output_ids
        del inputs
        del attention_mask
        del imgs
        # breakpoint()
        # if "Qwen" in model_args.model_name_or_path:
        #     filter_id = "\nassistant\n"
        #     for pred_sentence, gold in zip(pred_sentences, golds):
        #         pred_sentence = pred_sentence.strip()
        #         index = pred_sentence.find(filter_id)
        #         pred_sentence = pred_sentence[index + len(filter_id):]
        #         input_label = tokenizer.encode(gold)
        #         output_id = tokenizer.encode(pred_sentence)
        #         n_word = len(set(input_label))
        #         n_generated_word = len(set(output_id))
        #         n_correct = matching_token_num(output_id, input_label)
        #         predictions.append({"sentence":pred_sentence, "gt_sentence":gold.strip()})
        # else:
        for pred_sentence, gold in zip(pred_sentences, golds):
            pred_sentence = pred_sentence.strip()
            pred_sentence = pred_sentence[:len(gold)]
            
            input_label = tokenizer.encode(gold)
            output_id = tokenizer.encode(pred_sentence)
            n_word = len(set(input_label))
            n_generated_word = len(set(output_id))
            n_correct = matching_token_num(output_id, input_label)
            predictions.append({"sentence":pred_sentence, "gt_sentence":gold.strip()})
            
            
        n_word_total += n_word
        n_generated_word_total += n_generated_word
        n_word_correct += n_correct
        cnt += 1

    scores = NLPEvaluator(predictions).evaluate()
    scores["precision"] = n_word_correct / n_word_total
    scores["recall"] = n_word_correct / n_generated_word_total
    
    predictions.append(scores)
    #save predictions
    logger.info(f"Test | Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
    with open(f"./eval_results/{training_args.mode}/2025_{training_args.note}/round{round}_{dataname}.json", 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()

def evaluate_choices(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=2, choices=None):
    # if 'llava' in model_args.model_name_or_path.lower():
    #     conv = conversation_lib_llava.default_conversation
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
    # elif 'qwen' in model_args.model_name_or_path.lower():
    #     conv = conversation_lib_llava.default_conversation
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))

    conv = conversation_lib_llava.default_conversation
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=LLM_DataCollatorForGenerationDataset(tokenizer))
    
    repeat_criteria = CustomStoppingCriteria()
    stop_str = conv.sep2
    keywords = [stop_str]
    
    # img_feat_size = 729
    model.eval()
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        # for i, (inputs, imgs, golds, prompts, img_files) in enumerate(tqdm(dataloader)):
        for i, batch in enumerate((dataloader)): #tqdm
            # if "Qwen" in model_args.model_name_or_path:
            imgs = None
            inputs, golds, prompts = batch['input_ids'], batch['gold'], batch['prompt']
            attention_mask = batch['attention_mask'].to(device=device)
                
            inputs = inputs.to(device=device, non_blocking=True)
            # keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            # stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            # if "Qwen" in model_args.model_name_or_path:
            #     with torch.inference_mode():
            #         output_ids = model.generate(
            #             inputs[0],
            #             attention_mask=attention_mask[0],
            #             do_sample=True,# if args.temperature > 0 else False,
            #             temperature=training_args.eval_temp,#args.temperature,
            #             top_p=None,#args.top_p,
            #             num_beams=1,#args.num_beams,
            #             max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
            #             use_cache=True,
            #             pad_token_id=tokenizer.eos_token_id,
            #             stopping_criteria = stopping_criteria,
            #             # prompt=prompts if training_args.is_prompt else None,
            #         )
            # else:
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    do_sample=True,# if args.temperature > 0 else False,
                    temperature=training_args.eval_temp,#args.temperature,
                    top_p=None,#args.top_p,
                    num_beams=1,#args.num_beams,
                    max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    # stopping_criteria = stopping_criteria,
                )
            # if 'bunny' in model_args.model_name_or_path.lower():
            #     input_token_len = inputs.shape[1]
            #     output_ids = output_ids[:,input_token_len:]
            
            input_token_len = inputs.shape[1]
            if input_token_len < len(output_ids[0]):
                output_ids = output_ids[:,input_token_len:]
            
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
            # print("pred_sentences", pred_sentences)
            # print("prompts", prompts)
            del output_ids
            del inputs
            del attention_mask
            del imgs
            # if "Qwen" in model_args.model_name_or_path:
            #     filter_id = "\nassistant\n"
            #     for pred_sentence, prompt, gold in zip(pred_sentences, prompts, golds):
            #         pred_sentence = pred_sentence.strip()
                    
            #         index = pred_sentence.find(filter_id)
            #         pred_sentence = pred_sentence[index + len(filter_id):]
            #         # choices = parse_choice_list(prompt)
                    
            #         pred_option = can_infer(pred_sentence, choices)
                
            #         if isinstance(pred_option, str):
            #             # if gold == pred_option:
            #             if gold.lower() == pred_option.lower():
            #                 correct += 1
            #                 status='correct'
            #             else:
            #                 status='wrong'
            #         else:
            #             status = 'unkown'
            #         total += 1
            #         predictions.append({"sentence":pred_sentence, "gt_sentence":gold.strip(), 'status':status})
            # else:
            for pred_sentence, prompt, gold in zip(pred_sentences, prompts, golds):
                pred_sentence = pred_sentence.strip()
                
                # choices = parse_choice_list(prompt)
                
                # pred_option = can_infer(pred_sentence, choices)
                
            
                # if isinstance(pred_option, str):
                    # if gold == pred_option:
                if gold.lower() == pred_sentence[:len(gold)].lower():
                    correct += 1
                    status='correct'
                else:
                    status='wrong'
                # else:
                #     status = 'unkown'
                total += 1
                print("script", pred_sentence, gold.strip(), status)
                predictions.append({"sentence":pred_sentence, "gt_sentence":gold.strip(), 'status':status})

        scores = {'accuracy': correct/total}
        
        predictions.append(scores)
        #save predictions
        logger.info(f"Test | Data {dataname} | accuracy {scores['accuracy']} |")
        with open(f"./eval_results/{training_args.mode}/2025_{training_args.note}/round{round}_{dataname}.json", 'w') as fp:
            json.dump(predictions, fp, indent=4)
        torch.cuda.empty_cache()

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
        if all(choice.startswith("Image ") for choice in choices):
            choices = [re.match(r'(Image [A-D])', choice).group(1) for choice in choices]
        return choices
    
    match = re.search(r'Choice List: \[(.*?)\]', input_string)
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
    
    # If no choices found, return an empty list
    return []

def can_infer(answer, choices):
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
        (ModelArguments, DataArguments, TrainingArguments))
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

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"eval_results/{training_args.mode}/2025_{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'eval_results/{training_args.mode}/2025_{training_args.note}/round_{training_args.round_to_eval}.log', mode="w")

    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}')

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

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    
    # if "Qwen" in model_args.model_name_or_path:
    #     model, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    #     tokenizer = processor.tokenizer
    # else:
    #     model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    model, processor, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.setup, training_args.scenario)
    
    batch_size = 2 if 'l2p' in training_args.mode or 'dap' in training_args.mode or 'LAE' in training_args.mode else 1
    
    logger.info(f'Evaluating model at round {training_args.round_to_eval}')
    start_time = time.time()
    if not training_args.zeroshot:
        state_dir = f"{training_args.state_dir}_{training_args.note}"
        checkpoint_path = os.path.join(state_dir, f"server_model_round{training_args.round_to_eval-1}.pth")
        logger.info(f'load {checkpoint_path}')
        server_state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if True:
        if not training_args.zeroshot:
            model.load_state_dict(server_state_dict, strict=False)
        for round_i, data_info in enumerate(test_datalist):
            if round_i + 1 > training_args.round_to_eval:
                continue
            print(data_info['data_name'])
            # if "Qwen" in model_args.model_name_or_path:
            #     dataset = Qwen_GenerationDataset(data_info['data'], processor,tokenizer, data_args)
            # else:
            #     dataset = GenerationDataset(data_info['data'], tokenizer, data_args)
            dataset = LLM_GenerationDataset(data_info['data'], tokenizer, data_args, processor)
            if data_info['type'] == 'open-ended':
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size)
            elif data_info['type'] == 'multi-choice':
                dataset_name = data_info['data_name'].split('-')[0]
                dataset_subset_id = int(data_info['data_name'].split('-')[1])
                if dataset_name in list(CLASSIFICAION.keys()):
                    choices = CLASSIFICAION[dataset_name][dataset_subset_id]
                evaluate_choices(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, choices=choices)
            else:
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size)
        return
    
def get_datalists(args, setup, scenario_num):
    with open(f"./scenarios/{setup}/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    rounds_per_task = args.num_rounds
    task_data = scenario[0]
    train_datalist = []
    test_datalist = []
    eval_cnt = 0
    train_cnt = 0
    for data in task_data['datasets']:
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            for i in range(rounds_per_task):
                train_datalist.append(
                    {'datalist':datalist[i*samplenum_per_rounds:(i+1)*samplenum_per_rounds],
                     'train_cnt': train_cnt + samplenum_per_rounds})
                train_cnt += samplenum_per_rounds
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "type": data['type'] if 'type' in data else 'open-ended',
                "data": datalist[:min(len(datalist), 500)],
                "eval_cnt": eval_cnt})
            eval_cnt += len(datalist)
            
    return train_datalist, test_datalist

if __name__ == "__main__":
    main()
