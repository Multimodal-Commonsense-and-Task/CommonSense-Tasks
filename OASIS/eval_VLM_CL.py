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

from utils.data_loader_VLM import GenerationDataset, Qwen_GenerationDataset, DataCollatorForGenerationDataset, Qwen_DataCollatorForGenerationDataset
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
from PIL import Image
from utils.data_loader_VLM import dynamic_preprocess, IMG_END_TOKEN, IMAGENET_MEAN, IMAGENET_STD
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

ALPHABET = ['A','B','C','D','E','F']
DO_DYNAMIC_PROCESS=False


def patch_qwen_flash_attention_rotary_dtype():
    """Make Qwen 2.5-VL's FlashAttention rotary inputs dtype-compatible.

    The installed Transformers implementation promotes q/k to float32 before
    applying rotary embeddings, while a bf16-loaded vision encoder produces a
    bf16 rotary cache.  FlashAttention requires both operands to have the same
    dtype.  Keeping the cache float32 for this helper preserves its intended
    computation and the helper casts q/k back to their original dtype before
    FlashAttention is called.
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_vl

    if getattr(qwen_vl, "_coreset_rotary_dtype_patch", False):
        return

    original = qwen_vl.apply_rotary_pos_emb_flashatt

    def compatible_rotary(q, k, cos, sin):
        return original(q, k, cos.float(), sin.float())

    qwen_vl.apply_rotary_pos_emb_flashatt = compatible_rotary
    qwen_vl._coreset_rotary_dtype_patch = True

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


def load_image_internvl(image_file, input_size=448, max_num=12, do_dynamic_process=DO_DYNAMIC_PROCESS):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if do_dynamic_process:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def build_transform(input_size=448):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def evaluate(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=1, data=None):
    
    
    if "internvl" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = model.eval().cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        results_list = []
        imids = []
        acc = []
        for d in data:
            cnt += 1
            if 'image' in d:
                pixel_values_list = []
                num_patches_list = []
                images = d['image']
                
                # img_file = images[-1].split("/")[-1].split("_")[-1]
                if not isinstance(images, list):
                    images = [images]
                    
                for image in images:
                    pixel_values = load_image_internvl(image, max_num=12).to(torch.bfloat16).cuda()
                    pixel_values_list.append(pixel_values)
                    num_patches_list.append(pixel_values.size(0))
                pixel_values = torch.cat(pixel_values_list, dim=0)  
            question = d["conversations"][0]['value']
            
            with torch.inference_mode():
                response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list,
                    history=None, return_history=True)
            org_gold = d["conversations"][1]['value']
            pred_sentence = response.strip().lower()
            org_gold = org_gold.strip().lower()
            
            input_label = tokenizer.encode(org_gold)
            output_id = tokenizer.encode(pred_sentence)
            n_word = len(set(input_label))
            n_generated_word = len(set(output_id))
            n_correct = matching_token_num(output_id, input_label)
            
            n_word_total += n_word
            n_generated_word_total += n_generated_word
            n_word_correct += n_correct
            predictions.append({"question":question,"sentence":pred_sentence, "gt_sentence":org_gold})
                
    
    else:
        if 'llava' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
        elif 'qwen' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))

        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]
        
        # img_feat_size = 729
        predictions = []
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        cnt = 0
        # for i, (inputs, imgs, golds, prompts, img_files) in enumerate(tqdm(dataloader)):
        for i, batch in enumerate((dataloader)): #tqdm
            if "Qwen" in model_args.model_name_or_path:
                inputs, imgs, golds, img_files, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['image_grid_thw']
                attention_mask = batch['attention_mask'].to(device=device)[0]
            else:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            
            inputs = inputs.to(device=device, non_blocking=True)
            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            
            if "Qwen" in model_args.model_name_or_path:
                with torch.inference_mode():
                    output_ids = model.generate(
                        inputs[0],
                        attention_mask=attention_mask,
                        pixel_values=imgs,
                        image_grid_thw=image_grid_thw[0].cuda(),
                        # image_sizes=image_sizes,
                        do_sample=True,# if args.temperature > 0 else False,
                        temperature=training_args.eval_temp,#args.temperature,
                        top_p=None,#args.top_p,
                        num_beams=1,#args.num_beams,
                        max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria = stopping_criteria,
                        # prompt=prompts if training_args.is_prompt else None,
                    )
            else:
                with torch.inference_mode():
                    output_ids = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        images=imgs,
                        # image_sizes=image_sizes,
                        do_sample=True,# if args.temperature > 0 else False,
                        temperature=training_args.eval_temp,#args.temperature,
                        top_p=None,#args.top_p,
                        num_beams=1,#args.num_beams,
                        max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria = stopping_criteria,
                        # prompt=prompts if training_args.is_prompt else None,
                    )
            # if 'bunny' in model_args.model_name_or_path.lower():
            #     input_token_len = inputs.shape[1]
            #     output_ids = output_ids[:,input_token_len:]
            
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
            # breakpoint()
            if "Qwen" in model_args.model_name_or_path:
                filter_id = "\nassistant\n"
                for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                    pred_sentence = pred_sentence.strip()
                    
                    index = pred_sentence.find(filter_id)
                    pred_sentence = pred_sentence[index + len(filter_id):]
                    # print("pred_sentence", pred_sentence)
                    # print("gold", gold)
                    input_label = tokenizer.encode(gold)
                    output_id = tokenizer.encode(pred_sentence)
                    n_word = len(set(input_label))
                    n_generated_word = len(set(output_id))
                    n_correct = matching_token_num(output_id, input_label)
                    # print(pred_sentence)
                    predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip()})
            else:
                for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                    pred_sentence = pred_sentence.strip()
                    input_label = tokenizer.encode(gold)
                    output_id = tokenizer.encode(pred_sentence)
                    n_word = len(set(input_label))
                    n_generated_word = len(set(output_id))
                    n_correct = matching_token_num(output_id, input_label)
                    # print(pred_sentence)
                    predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip()})
                
                
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
    output_name = f"zeroshot_round{round}_{dataname}.json" if training_args.zeroshot else f"round{round}_{dataname}.json"
    with open(f"./eval_results/{training_args.mode}/2025_{training_args.note}/{output_name}", 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()

def evaluate_choices(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=2, data=None):
    if "internvl" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        total = 0
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = model.eval().cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        results_list = []
        imids = []
        acc = []
        for d in data:
            cnt += 1
            if 'image' in d:
                pixel_values_list = []
                num_patches_list = []
                images = d['image']
                img_file = images[-1].split("/")[-1].split("_")[-1]
                for image in images:
                    pixel_values = load_image_internvl(image, max_num=12).to(torch.bfloat16).cuda()
                    pixel_values_list.append(pixel_values)
                    num_patches_list.append(pixel_values.size(0))
                pixel_values = torch.cat(pixel_values_list, dim=0)  
            question = d["conversations"][0]['value']
            
            with torch.inference_mode():
                response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list,
                    history=None, return_history=True)
            org_gold = d["conversations"][1]['value']
            pred_sentence = response.strip().lower()
            org_gold = org_gold.strip().lower()
            
            # if "classification" in dataname:
            #     choices = CHOICES["classification"]
            # elif "nlvr2" in dataname:
            #     choices = CHOICES["nlvr2"]
            # else:
            choices = parse_choice_list(question)
            
            pred_option = can_infer(pred_sentence, choices)
        
            if isinstance(pred_option, str):
                # if gold == pred_option:
                if org_gold.lower() == pred_option.lower():
                    correct += 1
                    status='correct'
                else:
                    status='wrong'
            else:
                status = 'unkown'
            total += 1
            predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":org_gold.strip(), 'status':status})
    
    else:
    
        if 'llava' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
        elif 'qwen' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))
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
                if "Qwen" in model_args.model_name_or_path:
                    inputs, imgs, prompts, golds, img_files, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['prompt'], batch['gold'], batch['image_file'], batch['image_grid_thw']
                    attention_mask = batch['attention_mask'].to(device=device)[0]
                else:
                    inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
                    attention_mask = batch['attention_mask'].to(device=device)
                    
                inputs = inputs.to(device=device, non_blocking=True)
                if imgs is not None:
                    if isinstance(imgs, list):
                        imgs = [img.to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True) for img in imgs]
                    else:
                        imgs = imgs.to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True)
                    image_sizes = [x.shape[-2:] for x in imgs]
                keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
                stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
                if "Qwen" in model_args.model_name_or_path:
                    with torch.inference_mode():
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].cuda(),
                            # image_sizes=image_sizes,
                            do_sample=True,# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                else:
                    with torch.inference_mode():
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            # image_sizes=image_sizes,
                            do_sample=True,# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                # if 'bunny' in model_args.model_name_or_path.lower():
                #     input_token_len = inputs.shape[1]
                #     output_ids = output_ids[:,input_token_len:]
                
                pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
                # print("pred_sentences", pred_sentences)
                # print("prompts", prompts)
                del output_ids
                if "Qwen" in model_args.model_name_or_path:
                    filter_id = "\nassistant\n"
                    for pred_sentence, prompt, gold, img_file in zip(pred_sentences, prompts, golds, img_files):
                        pred_sentence = pred_sentence.strip()
                        
                        index = pred_sentence.find(filter_id)
                        pred_sentence = pred_sentence[index + len(filter_id):]
                        choices = parse_choice_list(prompt)
                        
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
                        predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip(), 'status':status})
                else:
                    for pred_sentence, prompt, gold, img_file in zip(pred_sentences, prompts, golds, img_files):
                        pred_sentence = pred_sentence.strip()
                        
                        choices = parse_choice_list(prompt)
                        
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
                        predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip(), 'status':status})

    scores = {'accuracy': correct/total}
    
    predictions.append(scores)
    #save predictions
    logger.info(f"Test | Data {dataname} | accuracy {scores['accuracy']} |")
    output_name = f"zeroshot_round{round}_{dataname}.json" if training_args.zeroshot else f"round{round}_{dataname}.json"
    with open(f"./eval_results/{training_args.mode}/2025_{training_args.note}/{output_name}", 'w') as fp:
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
    
    if "Qwen" in model_args.model_name_or_path:
        patch_qwen_flash_attention_rotary_dtype()
        if training_args.bits in [4, 8]:
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["vision"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                )
            ))
    else:
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

    if "Qwen" in model_args.model_name_or_path:
        model, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
        tokenizer = processor.tokenizer
    else:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.setup, training_args.scenario)
    
    batch_size = 2 if 'l2p' in training_args.mode or 'dap' in training_args.mode or 'LAE' in training_args.mode else 1
    
    logger.info(f'Evaluatiing server at round {training_args.round_to_eval}')
    start_time = time.time()
    server_eval_key = []
    
    if not training_args.zeroshot and training_args.eval_server:
        checkpoint_path = training_args.checkpoint_path or (
            f'./model_states_{training_args.note}/server_model_round{training_args.round_to_eval-1}.pth'
        )
        logger.info(f'load {checkpoint_path}')
        server_state_dict = torch.load(checkpoint_path, map_location='cpu')
        
    if training_args.eval_server and training_args.unseen_task:
        model.load_state_dict(server_state_dict, strict=False)
        # Qwen LoRA layers are created after the base model is cast and thus
        # default to float32.  FlashAttention requires its query/key/value
        # tensors to be fp16 or bf16 and to match the rotary cache dtype.
        # Recast after adapter construction and checkpoint loading.
        if "Qwen" in model_args.model_name_or_path and training_args.bf16:
            model.to(torch.bfloat16)
        for task_i, data_info in enumerate(test_datalists):
            # if task_i + 1 > training_args.round_to_eval:
            #     continue
            print(data_info['data_name'])
            if "Qwen" in model_args.model_name_or_path:
                dataset = Qwen_GenerationDataset(data_info['data'], processor,tokenizer, data_args)
            else:
                dataset = GenerationDataset(data_info['data'], tokenizer, data_args)
            if data_info['type'] == 'open-ended':
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, data=data_info['data'])
            elif data_info['type'] == 'multi-choice':
                evaluate_choices(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, data_info['data'])
            else:
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size)
        return
    
    logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    
def get_datalists(args, setup, scenario_num):
    with open(f"./scenarios/{setup}/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)

    task_data = scenario[0]
    train_datalist = []
    test_datalist = []
    eval_cnt = 0
    train_cnt = 0
    for data in task_data['datasets']:
        try:
            with open(f"./datalist/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
        except:
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
        test_datalist.append({
            "data_name": f"{data['dataset']}-{data['subset_id']}",
            "type": data['type'] if 'type' in data else 'open-ended',
            "data": datalist,
            "eval_cnt": eval_cnt})
        eval_cnt += len(datalist)
        

    return train_datalist, test_datalist

if __name__ == "__main__":
    main()
