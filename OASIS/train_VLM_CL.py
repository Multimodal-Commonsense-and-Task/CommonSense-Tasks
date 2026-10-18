import logging.config
import os
import random
import shutil
from scipy.special import softmax

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_task_vectors, load_deepspeed

from selection_methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset, Qwen_LazySupervisedDataset, Qwen_DataCollatorForSupervisedDataset, Intern_LazySupervisedDataset, Intern_DataCollatorForSupervisedDataset
from typing import Dict

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

from models.coda_prompt import CodaPrompt
from collections import OrderedDict

import json
import bisect
from collections import defaultdict
import gc


def is_budget_mode(mode):
    return "Budget" in mode or mode == "orin"


def is_gradsim_mode(mode):
    return "GradSim" in mode or mode == "oasis"


def remove_optimizer_checkpoints(state_dir, logger=logging.getLogger()):
    """Remove optimizer/scheduler artifacts after a successful final round.

    Only resume-state is deleted: the per-client DeepSpeed snapshots written by
    ``save_optim`` (``<state_dir>/client_<id>/``) plus any stray optimizer,
    scheduler or scaler files. The aggregated ``server_model_round*.pth``
    checkpoints and ``trainer_state.json`` are kept.
    """
    optimizer_filenames = {"optimizer.pt", "scheduler.pt", "scaler.pt"}
    removed_paths = []

    if not os.path.isdir(state_dir):
        return removed_paths

    # Whole per-client DeepSpeed resume snapshots (optimizer + partitioned
    # model shards live together and are useless without each other).
    for entry in sorted(os.listdir(state_dir)):
        path = os.path.join(state_dir, entry)
        if not (entry.startswith("client_") and os.path.isdir(path)):
            continue
        contents = os.listdir(path)
        is_ds_checkpoint = "latest" in contents or any(
            name.startswith("global_step") for name in contents
        )
        if not is_ds_checkpoint:
            continue
        try:
            shutil.rmtree(path)
            removed_paths.append(path)
        except OSError as e:
            logger.warning("Could not remove optimizer snapshot %s: %s", path, e)

    # Stray optimizer/scheduler files left anywhere under the state dir.
    for root, _, filenames in os.walk(state_dir):
        for filename in filenames:
            if filename in optimizer_filenames or filename.endswith("_optim_states.pt"):
                path = os.path.join(root, filename)
                try:
                    os.remove(path)
                    removed_paths.append(path)
                except OSError as e:
                    logger.warning("Could not remove optimizer file %s: %s", path, e)

    return removed_paths


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    
    if "Qwen" in model_args.model_name_or_path:
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

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    if "Qwen" in model_args.model_name_or_path:
        model, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    else:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)

    train_datalists, datasets_name = get_datalists(training_args, training_args.setup, training_args.scenario)
    
    # select functions
    set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    
    
    if training_args.resume_train_round > 0 and training_args.load_checkpoint is not None:
        logger.info(f'load {training_args.load_checkpoint}')
        server_state_dict = torch.load(training_args.load_checkpoint, map_location='cpu')
        
        with torch.no_grad():
            model.load_state_dict(server_state_dict, strict=False)

    global_state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    global_state_dict.update(non_lora_state_dict)
    
    current_task_vectors = None
    
    local_state_dict_list = [copy.deepcopy(global_state_dict)]
    old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
    local_state_dict_keys = local_state_dict_list[0].keys()
    extra_state_dict_dict = set_state_dict(model, global_state_dict, local_state_dict_list, training_args)
    training_loss = [[]]
    
    start_time = time.time()
    
    memory = [[]]
    memory_size = training_args.memory_size

    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    
    total_rounds = training_args.num_tasks
    
    lr_step = (init_lr - final_lr)/total_rounds
    mm_lr_step = (mm_init_lr - mm_final_lr)/total_rounds

    curr_ema = None
    curr_ema_var = 0
    cul_sub_dataset = []
    info_per_round = {}
    selected_info_per_round = {}
    selected_info_round_per_round = {}
    selected_info_ind_per_round = {}
    task_samples_num = []
    prev_all_inputs = None
    prev_all_input_inds = None
    
    klasses = np.array([])
    memory_count = [np.array([])]
    datalist = []
    datalist_round = []
    datalist_prob = []
    ema_pool = []
    
    memory_info_list = []
    
    # updated_loss = []
    # retrieved_index = []
    for curr_round in range(total_rounds):
        old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
        
        selected_ids = [0]
        num_selection = 1
        training_args.learning_rate = init_lr - lr_step*curr_round
        training_args.mm_projector_lr = mm_init_lr - mm_lr_step*curr_round
        if curr_round > 0 and training_args.is_wsd:
            training_args.warmup_ratio = 0
            training_args.warmup_steps = 0
        for r_i, idx in enumerate(range(num_selection)):
            model.config.use_cache = False
            torch.cuda.empty_cache()
            client_id = selected_ids[idx]     
            
            ##### simulate online memory insertion & get_batch ####
            sub_dataset = train_datalists[curr_round]['datalist']
            if training_args.seed > 1:
                random.shuffle(sub_dataset)
            task_sub_dataset_length = len(sub_dataset)
            num_iterations = train_datalists[curr_round]['num_iter']

            sample_ratio = training_args.sample_ratio
            
            if curr_round > 0:
                cul_task_num = task_samples_num[-1]+len(sub_dataset)
                task_samples_num.append(cul_task_num)
            else:
                task_samples_num.append(len(sub_dataset)-1)
                
            logger.info(f"original datalist {len(sub_dataset)}")
            model.zero_grad()
            
            task_id = train_datalists[curr_round]['task_id']
            
            extra_state_dict_dict['client_id'] = client_id
            extra_state_dict_dict['curr_round'] = curr_round
            if training_args.use_task_id:
                extra_state_dict_dict['task_id'] = task_id
            
            load_state_dict(model, global_state_dict, old_local_state_dict_list, client_id, training_args, extra_state_dict_dict)
            print('model loading done')
                
            iteration = 0
            datalist = []
            datalist_round = []
            if not training_args.is_streamonly:
                # memory-only
                for i, sample in enumerate(sub_dataset):
                    memory[client_id].append(sample)
                    iteration += num_iterations
                    if iteration >= 1:
                        for _ in range(int(iteration)):
                            
                            # if training_args.selection_method == "random":
                            #     if total_batchsize >=2:
                            #         total_batchsize = int(total_batchsize/2)
                            
                            batch = random.sample(memory[client_id], k=min(len(memory[client_id]), total_batchsize))
                            mul = (total_batchsize//len(batch)) + 1
                            batch = (batch*mul)[:total_batchsize]
                            
                            datalist.extend(batch[:])
                            iteration -= 1
                            for batch_data in batch:
                                # image = batch_data["image"]
                                # if isinstance(image, list):
                                #     image = image[0]
                                # image_data = image.split("/")[1]
                                # position = datasets_name.index(image_data)
                                index = memory[client_id].index(batch_data)
                                position = bisect.bisect_left(task_samples_num, index)
                                datalist_round.append(position)
                                # datalist_prob.append(sample_score[index])
                                # retrieved_index.append(index)
                                
                
                if len(datalist) < num_iterations*total_batchsize:
                    batch = random.sample(memory[client_id], k=min(len(memory[client_id]), total_batchsize))
                    mul = (total_batchsize//len(batch)) + 1
                    batch = (batch*mul)[:total_batchsize]
                    datalist.extend(batch[:])
                    for batch_data in batch:
                        # image = batch_data["image"]
                        # if isinstance(image, list):
                        #     image = image[0]
                        # image_data = image.split("/")[1]
                        # position = datasets_name.index(image_data)
                        index = memory[client_id].index(batch_data)
                        position = bisect.bisect_left(task_samples_num, index)
                        datalist_round.append(position)
                        # datalist_prob.append(sample_score[index])
                        
            elif training_args.is_sar:
                print("@@SAR@@")
                ### aL-SAR ###
                T = 0.125
                k_coeff = 0.4
                count_decay_ratio = 0.9

                for i, sample in enumerate(sub_dataset):
                    if len(memory[client_id]) == memory_size:
                        pop_index = random.randrange(memory_size)
                        memory[client_id].pop(pop_index)
                        memory_count[client_id] = np.delete(memory_count[client_id], pop_index, 0)
                    
                    memory[client_id].append(sample)
                    memory_count[client_id] = np.append(memory_count[client_id], 0)
                    iteration += num_iterations
                    if iteration >= 1:
                        for _ in range(int(iteration)):
                            # if len(memory[client_id]) > total_batchsize:
                                # count_decay_ratio = total_batchsize / (len(memory[client_id])*k_coeff)
                            memory_count[client_id] *= count_decay_ratio 
                            sample_score = memory_count[client_id]
                            weight = softmax(-sample_score/T)
                            sample_idx = np.random.choice(len(memory[client_id]), min(len(memory[client_id]), total_batchsize), p=weight, replace=False)
                            batch = [memory[client_id][idx] for idx in sample_idx]
                            mul = (total_batchsize//len(batch)) + 1
                            batch = (batch*mul)[:total_batchsize]
                            datalist.extend(batch[:])
                            iteration -= 1
                            for idx in sample_idx:
                                memory_count[client_id][idx] += 1
                            for batch_data in batch:
                                # index = memory[client_id].index(batch_data)
                                # position = bisect.bisect_left(task_samples_num, index)
                                # datalist_round.append(position)
                                # datalist_prob.append(sample_score[index])
                                image = batch_data["image"]
                                if isinstance(image, list):
                                    image = image[0]
                                image_data = image.split("/")[1]
                                position = datasets_name.index(image_data)
                                index = memory[client_id].index(batch_data)
                                # position = bisect.bisect_left(task_samples_num, index)
                                datalist_round.append(position)
                                datalist_prob.append(sample_score[index])
                if len(datalist) < num_iterations*total_batchsize:
                    memory_count[client_id] *= count_decay_ratio
                    sample_score = memory_count[client_id]
                    weight = softmax(-sample_score/T)
                    sample_idx = np.random.choice(len(memory[client_id]), min(len(memory[client_id]), total_batchsize), p=weight, replace=False)
                    batch = [memory[client_id][idx] for idx in sample_idx]
                    mul = (total_batchsize//len(batch)) + 1
                    batch = (batch*mul)[:total_batchsize]
                    datalist.extend(batch[:])
                    for idx in sample_idx:
                        memory_count[client_id][idx] += 1
                    for batch_data in batch:
                        # index = memory[client_id].index(batch_data)
                        # position = bisect.bisect_left(task_samples_num, index)
                        # datalist_round.append(position)
                        # datalist_prob.append(sample_score[index])
                        image = batch_data["image"]
                        if isinstance(image, list):
                            image = image[0]
                        image_data = image.split("/")[1]
                        position = datasets_name.index(image_data)
                        index = memory[client_id].index(batch_data)
                        # position = bisect.bisect_left(task_samples_num, index)
                        datalist_round.append(position)
                        datalist_prob.append(sample_score[index])
                        
            elif training_args.is_sar_task:
                ### aL-SAR ###
                T = 0.125
                k_coeff = 0.4
                count_decay_ratio = 0.9

                for i, sample in enumerate(sub_dataset):
                    if len(memory[client_id]) == memory_size:
                        pop_index = random.randrange(memory_size)
                        memory[client_id].pop(pop_index)
                        memory_count[client_id] = np.delete(memory_count[client_id], pop_index, 0)
                    
                    memory[client_id].append(sample)
                    memory_count[client_id] = np.append(memory_count[client_id], 0)
                    iteration += num_iterations
                    if iteration >= 1:
                        for _ in range(int(iteration)):
                            # if len(memory[client_id]) > total_batchsize:
                                # count_decay_ratio = total_batchsize / (len(memory[client_id])*k_coeff)

                            memory_count[client_id] *= count_decay_ratio 
                            sample_score = memory_count[client_id]
                            if curr_round > 0:
                                if curr_round == 1:
                                    full_index  = i 
                                else:
                                    full_index = i+task_samples_num[curr_round-2]+1
                                for round_n in range(curr_round+1):
                                    if round_n == 0:
                                        sample_score[:task_samples_num[round_n]+1] *= mem_retrieve_comp1[round_n][full_index]
                                        sample_score[:task_samples_num[round_n]+1] *= sum(memory_count[client_id][:task_samples_num[round_n]+1])
                                    elif round_n == curr_round:
                                        sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n-1]+2+i] *= mem_retrieve_comp1[round_n][full_index]
                                        sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n-1]+2+i] *= sum(memory_count[client_id][task_samples_num[round_n-1]+1:task_samples_num[round_n-1]+2+i])
                                    else:
                                        sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n]+1] *= mem_retrieve_comp1[round_n][full_index]
                                        sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n]+1] *= sum(memory_count[client_id][task_samples_num[round_n-1]+1:task_samples_num[round_n]+1])
                            
                            weight = softmax(-sample_score/T)
                            sample_idx = np.random.choice(len(memory[client_id]), min(len(memory[client_id]), total_batchsize), p=weight, replace=False)
                            batch = [memory[client_id][idx] for idx in sample_idx]
                            mul = (total_batchsize//len(batch)) + 1
                            batch = (batch*mul)[:total_batchsize]
                            datalist.extend(batch[:])
                            iteration -= 1
                            for idx in sample_idx:
                                memory_count[client_id][idx] += 1
                            for batch_data in batch:
                                index = memory[client_id].index(batch_data)
                                position = bisect.bisect_left(task_samples_num, index)
                                datalist_round.append(position)
                                datalist_prob.append(sample_score[index])
                if len(datalist) < num_iterations*total_batchsize:
                    memory_count[client_id] *= count_decay_ratio
                    sample_score = memory_count[client_id]
                    if curr_round > 0:
                        if curr_round == 1:
                            full_index  = i 
                        else:
                            full_index = i+task_samples_num[curr_round-2]+1
                        for round_n in range(curr_round+1):
                            if round_n == 0:
                                sample_score[:task_samples_num[round_n]+1] *= mem_retrieve_comp1[round_n][full_index]
                                sample_score[:task_samples_num[round_n]+1] *= sum(memory_count[client_id][:task_samples_num[round_n]+1])
                            elif round_n == curr_round:
                                sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n-1]+2+i] *= mem_retrieve_comp1[round_n][full_index]
                                sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n-1]+2+i] *= sum(memory_count[client_id][task_samples_num[round_n-1]+1:task_samples_num[round_n-1]+2+i])
                            else:
                                sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n]+1] *= mem_retrieve_comp1[round_n][full_index]
                                sample_score[task_samples_num[round_n-1]+1:task_samples_num[round_n]+1] *= sum(memory_count[client_id][task_samples_num[round_n-1]+1:task_samples_num[round_n]+1])
                                
                    weight = softmax(-sample_score/T)
                    sample_idx = np.random.choice(len(memory[client_id]), min(len(memory[client_id]), total_batchsize), p=weight, replace=False)
                    batch = [memory[client_id][idx] for idx in sample_idx]
                    mul = (total_batchsize//len(batch)) + 1
                    batch = (batch*mul)[:total_batchsize]
                    datalist.extend(batch[:])
                    for idx in sample_idx:
                        memory_count[client_id][idx] += 1
                    for batch_data in batch:
                        index = memory[client_id].index(batch_data)
                        position = bisect.bisect_left(task_samples_num, index)
                        datalist_round.append(position)
                        datalist_prob.append(sample_score[index])
            else:
                # stream-only
                # datalist = sub_dataset[:num_iterations*total_batchsize]
                datalist = sub_dataset

            if training_args.resume_train_round > 0 and curr_round+1<training_args.resume_train_round:
                continue
                
            if "Qwen" in model_args.model_name_or_path:
                data_module = make_qwen_supervised_data_module(client_data=datalist, # sub_dataset
                                    tokenizer=processor,
                                    data_args=copy.deepcopy(data_args), model_name=model_args.model_name_or_path, curr_round=curr_round)
            else:
                if training_args.mode == "perturb":
                    data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                        tokenizer=tokenizer,
                        data_args=copy.deepcopy(data_args), model_name=model_args.model_name_or_path, curr_round=curr_round, config=model.config, perturb_num=3)
                else:
                    data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                        tokenizer=tokenizer,
                        data_args=copy.deepcopy(data_args), model_name=model_args.model_name_or_path, curr_round=curr_round, config=model.config)
                        
            logger.info(f"check datalist {len(datalist)}")
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train | num samples {len(datalist)}')

            # ===== Train the continual-learning model =====
            if training_args.use_fisher:
                extra_state_dict_dict['fisher_old'] = fisher_olds[client_id]
                
            if training_args.use_task_vector:
                extra_state_dict_dict['task_vector'] = task_vectors[client_id]

            if "Qwen" in model_args.model_name_or_path and curr_round==0:
                tokenizer = processor.tokenizer
                
            if "sft_org" in training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob, ema_pool=ema_pool, model_args=model_args)
            if is_budget_mode(training_args.mode):
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob, ema_pool=ema_pool)
            elif is_gradsim_mode(training_args.mode):
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob, ema_pool=ema_pool)
            elif "Mutual" in training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob, ema_pool=ema_pool)
            elif "divbs" in training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob)
            elif "cluster" == training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)
            elif "infoBatch" == training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var)
            else:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)

            results = trainer.train()
            training_loss[client_id].append(results.training_loss)
            if training_args.use_fisher:
                fisher_olds[client_id] = trainer.fisher_old
            
            if training_args.use_task_vector:
                task_vectors[client_id] = trainer.task_vector #- original_weights
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                path = os.path.join(training_args.state_dir, "trainer_state.json")
                trainer.state.save_to_json(path)
            
            model.config.use_cache = True
            
            # save local model
            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                state_dict.update(non_lora_state_dict)
            else:
                state_dict = {k: t.detach().cpu().clone() for k, t in model.named_parameters() if t.requires_grad}
            
            local_state_dict_list[client_id] = copy.deepcopy(state_dict)

            k_to_del = []
            for k in state_dict.keys():
                if k not in local_state_dict_keys:
                    k_to_del.append(k)
            for k in k_to_del:
                del state_dict[k]

            local_state_dict = getattr(trainer, 'global_weight', None)
            if local_state_dict is not None:
                local_state_dict_list[client_id] = copy.deepcopy(local_state_dict)
            
            trainer.deepspeed.empty_partition_cache()
            
            if "cluster" == training_args.mode:
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                
            if is_gradsim_mode(training_args.mode):
                round_info = copy.deepcopy(trainer.round_info)
                selected_info = copy.deepcopy(trainer.selected_info)
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                round_info = [float(info) for info in round_info]
                selected_info = [float(info) for info in selected_info]
                print(f"selected samples in round {curr_round}:", len(selected_info_ind))
                selected_info_round = []
                for ind in selected_info_ind:
                    selected_info_round.append(datalist_round[ind])
                selected_inputs = [datalist[ind] for ind in selected_info_ind]
                info_per_round = round_info
                selected_info_per_round = selected_info
                selected_info_round_per_round = selected_info_round
                selected_info_ind_per_round = selected_info_ind
                curr_ema = copy.deepcopy(trainer.curr_ema)
                curr_ema_var = copy.deepcopy(trainer.curr_ema_var)
            
            if "Mutual" in training_args.mode:
                round_info = copy.deepcopy(trainer.round_info)
                selected_info = copy.deepcopy(trainer.selected_info)
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                round_info = [float(info) for info in round_info]
                selected_info = [float(info) for info in selected_info]
                print(f"selected samples in round {curr_round}:", len(selected_info_ind))
                selected_info_round = []
                for ind in selected_info_ind:
                    selected_info_round.append(datalist_round[ind])
                selected_inputs = [datalist[ind] for ind in selected_info_ind]
                info_per_round = round_info
                selected_info_per_round = selected_info
                selected_info_round_per_round = selected_info_round
                selected_info_ind_per_round = selected_info_ind
                curr_ema = copy.deepcopy(trainer.curr_ema)
                curr_ema_var = copy.deepcopy(trainer.curr_ema_var)
            
            if "score" == training_args.mode:
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
            
            if training_args.mode in ["divbs", "infoBatch", "gradnorm", "maxloss", "sft"]:
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
            
            if is_budget_mode(training_args.mode):
                round_info = copy.deepcopy(trainer.round_info)
                selected_info = copy.deepcopy(trainer.selected_info)
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                ema_list = copy.deepcopy(trainer.ema_list)
                round_info = [float(info) for info in round_info]
                selected_info = [float(info) for info in selected_info]
                selected_info_round = []
                for ind in selected_info_ind:
                    selected_info_round.append(datalist_round[ind])
                selected_inputs = [datalist[ind] for ind in selected_info_ind]
                info_per_round = round_info
                selected_info_per_round = selected_info
                selected_info_round_per_round = selected_info_round
                selected_info_ind_per_round = selected_info_ind
                curr_ema = copy.deepcopy(trainer.curr_ema)
                curr_ema_var = copy.deepcopy(trainer.curr_ema_var)
                
            trainer.accelerator.free_memory()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"done Round {curr_round} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    

        if training_args.resume_train_round > 0 and curr_round+1<training_args.resume_train_round:
            continue
        
        aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **extra_state_dict_dict)
        
        # Save server model
        if (training_args.local_rank == 0 or training_args.local_rank == -1): 
            torch.save(global_state_dict, os.path.join(training_args.state_dir, f"server_model_round{curr_round}.pth"))
            
        # print("info_per_round", info_per_round)
        if "sft" == training_args.mode:
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = datalist_round
            # selected_info_ind = []
            # for batch_data in datalist:
            #     index = memory[client_id].index(batch_data)
            #     selected_info_ind.append(index)
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
        
        # elif "sft_org" == training_args.mode:
        #     comb_info_per_round = {}
        #     comb_info_per_round["info_stats"] = info_stats
        #     with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
        #         json.dump(comb_info_per_round, all_info_per_round)
        
        
        if training_args.mode in ["divbs", "infoBatch", "gradnorm", "maxloss"]:
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = datalist_round
            # selected_info_ind = []
            # for batch_data in datalist:
            #     index = memory[client_id].index(batch_data)
            #     selected_info_ind.append(index)
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
        
        if "score" in training_args.mode:
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = datalist_round
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
        
        if "cluster" == training_args.mode:
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = datalist_round
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
        
        if is_budget_mode(training_args.mode):
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = info_per_round
            comb_info_per_round["selected_info_per_round"] = selected_info_per_round
            # comb_info_per_round["selected_inputs"] = selected_inputs
            comb_info_per_round["selected_info_round_per_round"] = selected_info_round_per_round
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            comb_info_per_round["ema_list"] =ema_list
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
                
        if is_gradsim_mode(training_args.mode):
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = info_per_round
            comb_info_per_round["selected_info_per_round"] = selected_info_per_round
            # comb_info_per_round["selected_inputs"] = selected_inputs
            comb_info_per_round["selected_info_round_per_round"] = selected_info_round_per_round
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            # comb_info_per_round["ema_list"] =ema_list
            # comb_info_per_round["calculated_cos_sim"] = task_cossim
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
                
        if "Mutual" in training_args.mode:
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = info_per_round
            comb_info_per_round["selected_info_per_round"] = selected_info_per_round
            # comb_info_per_round["selected_inputs"] = selected_inputs
            comb_info_per_round["selected_info_round_per_round"] = selected_info_round_per_round
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            # comb_info_per_round["ema_list"] =ema_list
            # comb_info_per_round["calculated_mutual"] = task_mutual
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
                
        trainer.deepspeed.empty_partition_cache()
        trainer.accelerator.free_memory()
        del trainer
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    # Server checkpoints are saved after each round's aggregation. Now that every
    # round finished successfully, drop the optimizer/scheduler resume state.
    if training_args.local_rank in (0, -1):
        removed_paths = remove_optimizer_checkpoints(training_args.state_dir, logger)
        if removed_paths:
            logger.info(
                "Removed %d optimizer checkpoint path(s): %s",
                len(removed_paths), ", ".join(removed_paths),
            )

    logger.info("total done\n")

def make_qwen_supervised_data_module(client_data, tokenizer,
                                data_args, model_name, curr_round) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Qwen_LazySupervisedDataset(client_data, tokenizer, data_args, model_id=model_name)
    data_collator = Qwen_DataCollatorForSupervisedDataset(tokenizer=tokenizer.tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_name, curr_round, config=None, perturb_num=0) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if "intern" in model_name.lower():
        train_dataset = Intern_LazySupervisedDataset(client_data, tokenizer, data_args, config=config)
        data_collator = Intern_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
    else:
        if perturb_num > 0:
            train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args, perturb_num=perturb_num)
        else:
            train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset,
                    eval_dataset=None,
                    data_collator=data_collator)

def get_datalists(args, setup, scenario_num):
    with open(f"./scenarios/{setup}/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)

    
    max_iterations = args.num_iter
    
    datasets_name = []

    task_data = scenario[0]
    train_datalist = []
    test_datalist = []
    eval_cnt = 0
    for task_id, data in enumerate(task_data['datasets']):
        datasets_name.append(data['dataset'])
        with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
            datalist = json.load(fp)
        num_iter = max_iterations
        train_datalist.append(
            {'datalist': datalist,
                'num_iter': num_iter,
                'task_id': task_id})
        

    return train_datalist, datasets_name

if __name__ == "__main__":
    main()
