import logging.config
import os
import random
from scipy.special import softmax

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_task_vectors, load_deepspeed

from selection_methods.method_manager import select_method
from utils.data_loader_VLM import LLM_LazySupervisedDataset, LLM_DataCollatorForSupervisedDataset
from typing import Dict

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

from models.coda_prompt import CodaPrompt
from collections import OrderedDict

# from utils.selection_methods import sample_selection
import json
import bisect
from collections import defaultdict
import gc

os.environ["WANDB_DISABLED"] = "true"

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

    model, processor, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
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
            
        # if training_args.mode == 'fedours':
        #     local_state_dict = {}
        #     for k in server_state_dict.keys():
        #         new_k = k.replace('ia3_l_1', 'ia3_l_2')
        #         local_state_dict[new_k] = server_state_dict[k]
            
        #     model.load_state_dict(local_state_dict, strict=False)
    
    global_state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    global_state_dict.update(non_lora_state_dict)
    
    # if training_args.fedours: #training_args.mode == 'fedours' and 
    #     logger.info(f'load task vector {training_args.load_checkpoint}')
    #     tv_weights = torch.load(training_args.load_checkpoint, map_location='cpu')
    #     prev_task_vectors = tv_weights['task_vectors']
    #     prev_local_state_dict_list = tv_weights['local_state_dict_list']
    #     current_task_vectors = get_task_vectors(model, tokenizer, train_datalists, training_args, data_args, global_state_dict, make_supervised_data_module)
    # else:
    current_task_vectors = None
    
    local_state_dict_list = [copy.deepcopy(global_state_dict)]
    old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
    local_state_dict_keys = local_state_dict_list[0].keys()
    extra_state_dict_dict = set_state_dict(model, global_state_dict, local_state_dict_list, training_args)
    # torch.save(global_state_dict, os.path.join(training_args.state_dir, f"server_model_base.pth"))
    # print("non_lora_state_dict", non_lora_state_dict)
    # print("prev_local_state_dict_keys", local_state_dict_keys)
    # print("global_state_dict", list(global_state_dict.keys()))
    training_loss = [[]]
    
    
    
    # Start single-stream continual learning.
    start_time = time.time()
    
    memory = [[]]
    memory_size = training_args.memory_size

    # if training_args.sample_ratio == 1:
    #     total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    # else:
    # if training_args.mode == "sft":
    #     total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    # else:   
    # if training_args.mode == "sft_org":
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    # else:   
    #     total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps*16
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    
    total_rounds = training_args.num_rounds * training_args.num_tasks
    fisher_olds = [None]
    task_vectors = [None]
    
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
            # with open("./importance/DISJOINT_WHEN7_NEURIPS_3_12_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_03125_fisherbudget_soft0.1_emaavg100_updateema_warmup32_scenario8_10000_sample0_0625_emaratio0_01_warmup32_z1.54_localrank0_round0.json", "r") as r_json:
            #     read_json = json.load(r_json)
            # sub_dataset = read_json["selected_inputs"]
            # sub_dataset = random.sample(sub_dataset, k=5000)
            if training_args.seed > 1:
                random.shuffle(sub_dataset)
            task_sub_dataset_length = len(sub_dataset)
            num_iterations = train_datalists[curr_round]['num_iter']
            # random.shuffle(sub_dataset)
            # if not training_args.is_streamonly:
            # cul_sub_dataset.extend(sub_dataset)
            # else:
            # cul_sub_dataset = sub_dataset
            # updated_loss.extend(torch.ones(len(sub_dataset)))
            # random.shuffle(cul_sub_dataset)
            sample_ratio = training_args.sample_ratio
            print("sample_ratio", sample_ratio)
            
            # # DEBUG
            # sub_dataset = sub_dataset[:50]
            
            
            if curr_round > 0:
                cul_task_num = task_samples_num[-1]+len(sub_dataset)
                task_samples_num.append(cul_task_num)
            else:
                task_samples_num.append(len(sub_dataset)-1)
                
                
            # train_sample_vectors = None
            # if training_args.selection_method not in ["no_sampling", "random"]:
            #     print("get_representations")
            #     # if curr_round != 0:
            #     train_sample_vectors = get_task_vectors(model, tokenizer, cul_sub_dataset, training_args, data_args, global_state_dict, make_supervised_data_module)
                
            #     # if training_args.selection_method != "info":
            #     np.save(f'embedding_{training_args.selection_method}_k{training_args.k_means}_round{curr_round}.npy', train_sample_vectors)
            #     # else:
            #     #     train_sample_vectors = np.load(f'embedding_{training_args.selection_method}_k{training_args.k_means}_round{curr_round}.npy')
            
            logger.info(f"original datalist {len(sub_dataset)}")
            # if training_args.selection_method != "info":
            # selected_dataset = sample_selection(model, cul_sub_dataset, training_args.selection_method, data_args=copy.deepcopy(data_args), training_args=training_args, model_args=model_args, tokenizer=tokenizer, sample_ratio=sample_ratio,train_sample_vectors=train_sample_vectors, k=training_args.k_means)
            # else:
                # selected_dataset, selected_info = sample_selection(model, cul_sub_dataset, training_args.selection_method, data_args=copy.deepcopy(data_args), training_args=training_args, model_args=model_args, tokenizer=tokenizer, sample_ratio=sample_ratio,train_sample_vectors=train_sample_vectors, k=training_args.k_means)
            model.zero_grad()
            
            task_id = train_datalists[curr_round]['task_id']
            
            extra_state_dict_dict['client_id'] = client_id
            extra_state_dict_dict['curr_round'] = curr_round
            if training_args.use_task_id:
                extra_state_dict_dict['task_id'] = task_id
            
            load_state_dict(model, global_state_dict, old_local_state_dict_list, client_id, training_args, extra_state_dict_dict)
            print('model loading done')
            
            if training_args.is_sar_task and curr_round==0:
                all_task_dataset = []
                sar_task_samples_num = []
                all_task_gradient_ema = defaultdict(list)
                all_task_ema_coeff = defaultdict(list)
                all_task_self_cls_sim = defaultdict(list)
                for r_i in range(total_rounds):
                    grad_sub_dataset = train_datalists[r_i]['datalist']
                    all_task_dataset.extend(grad_sub_dataset)
                    if r_i > 0:
                        cul_task_num = sar_task_samples_num[-1]+len(grad_sub_dataset)
                        sar_task_samples_num.append(cul_task_num)
                    else:
                        sar_task_samples_num.append(len(grad_sub_dataset)-1)
                data_module = make_supervised_data_module(client_data=all_task_dataset, # sub_dataset
                            tokenizer=tokenizer,
                            data_args=copy.deepcopy(data_args), model_name=model_args.model_name_or_path, curr_round=curr_round)
                clone_training_args = copy.deepcopy(training_args)
                clone_training_args.per_gpu_train_batch_size=1
                clone_training_args.gradient_accumulation_steps=1
                
                trainer = vectorretreive_create_trainer(model, tokenizer, clone_training_args, data_module, extra_state_dict_dict, sar_task_samples_num=sar_task_samples_num )
                results = trainer.train()
                # all_task_gradients = trainer.all_task_gradients
                mem_retrieve_comp1 = trainer.mem_retrieve_comp1
                
                del trainer
                    
                
            iteration = 0
            datalist = []
            datalist_round = []
            # iter_ratio = num_iterations / len(selected_dataset)
            if not training_args.is_streamonly:
                # memory-only
                for i, sample in enumerate(sub_dataset):
                    # if len(memory[client_id]) == memory_size:
                    #     memory[client_id].pop(random.randrange(memory_size))
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
                                # gradient_per_task = []
                                # ema_coeff_per_task = []
                                # self_cls_sim_per_task = []
                                # for round_n in range(curr_round+1):
                                #     if round_n < curr_round:
                                #         gradient_per_task.append(all_task_gradient_ema[curr_round][-1].flatten())
                                #         ema_coeff_per_task.append(all_task_ema_coeff[curr_round][-1])
                                #         self_cls_sim_per_task.append(all_task_ema_coeff[curr_round][-1])
                                #     else:
                                #         gradient_per_task.append(all_task_gradient_ema[curr_round][i].flatten())
                                #         ema_coeff_per_task.append(all_task_ema_coeff[curr_round][i])
                                #         self_cls_sim_per_task.append(all_task_self_cls_sim[curr_round][i])
                                # gradient_per_task = torch.stack(gradient_per_task)
                                # ema_coeff_per_task = torch.cat(ema_coeff_per_task)
                                # self_cls_sim_per_task = torch.cat(self_cls_sim_per_task)
                                # # cosine_sim_matrix = torch.mm(gradient_per_task, gradient_per_task.T)
                                # # norms = torch.norm(gradient_per_task, dim=1, keepdim=True)
                                # # cosine_sim_matrix /= (norms @ norms.T)
                                # cosine_sim_matrix = F.cosine_similarity(gradient_per_task.unsqueeze(1), gradient_per_task.unsqueeze(0), dim=2)
                                # cosine_sim_matrix *= torch.sqrt(self_cls_sim_per_task.unsqueeze(0) * self_cls_sim_per_task.unsqueeze(1))
                                # ema_coeff = ema_coeff_per_task.unsqueeze(0) * ema_coeff_per_task.unsqueeze(1)
                                # cosine_sim_matrix *= ema_coeff

                                # row_sums = torch.sum(cosine_sim_matrix, dim=1)
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
                # print("before", len(datalist))
                # print("after", len(sub_dataset[:num_iterations*total_batchsize]))
            
            if training_args.resume_train_round > 0 and curr_round+1<training_args.resume_train_round:
                continue
                
  
            data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                        processor=processor,
                        tokenizer=tokenizer,
                        data_args=copy.deepcopy(data_args))
                
            logger.info(f"check datalist {len(datalist)}")
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train | num samples {len(datalist)}')

            # ===== Train the continual-learning model =====
            if training_args.use_fisher:
                extra_state_dict_dict['fisher_old'] = fisher_olds[client_id]
                
            if training_args.use_task_vector:
                extra_state_dict_dict['task_vector'] = task_vectors[client_id]
            
            # for n, p in model.base_model.named_parameters():
            #     if p.requires_grad == True:
            #         print("train", n)
            # model.set_state('lora1')
            # model.activate_lora1()
            if "Qwen" in model_args.model_name_or_path and curr_round==0:
                tokenizer = processor.tokenizer
                
            if "Budget" in training_args.mode or "sft_org" in training_args.mode:
                # if training_args.start_ema_update_step is not None:
                #     ema_data_module = make_supervised_data_module(client_data=datalist[:((training_args.start_ema_update_step//total_batchsize)+1)], # sub_dataset
                #         tokenizer=tokenizer,
                #         data_args=copy.deepcopy(data_args))
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob, ema_pool=ema_pool)
            elif "GradSim" in training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob, ema_pool=ema_pool)
            elif "Mutual" in training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob, ema_pool=ema_pool)
            elif "divbs" in training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var, datalist_prob=datalist_prob)
            elif "cluster" == training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)
            elif "online" == training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, memory_info_list=memory_info_list, memory_samples=memory[client_id])
            elif "infoBatch" == training_args.mode:
                trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=curr_ema, curr_ema_var=curr_ema_var)
            elif "maxloss" == training_args.mode:
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
            # output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")
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
            # print("state_dict", list(state_dict.keys()))

            k_to_del = []
            for k in state_dict.keys():
                if k not in local_state_dict_keys:
                    k_to_del.append(k)
            for k in k_to_del:
                print("del", k)
                del state_dict[k]
            # if (training_args.local_rank == 0 or training_args.local_rank == -1):
            #     torch.save(state_dict, output_dir)
            
            local_state_dict = getattr(trainer, 'global_weight', None)
            if local_state_dict is not None:
                local_state_dict_list[client_id] = copy.deepcopy(local_state_dict)
            
            trainer.deepspeed.empty_partition_cache()
            
            if "cluster" == training_args.mode:
                # prev_all_inputs = copy.deepcopy(trainer.memory_inputs)
                # prev_all_input_inds = copy.deepcopy(trainer.memory_input_inds)
                # memory_input_inds = copy.deepcopy(trainer.memory_input_inds)
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                # selected_inputs = copy.deepcopy(trainer.selected_inputs)
                # real_selected_info_ind = [memory_input_inds[ind] for ind in selected_info_ind]
                
            if "GradSim" in training_args.mode:
                round_info = copy.deepcopy(trainer.round_info)
                selected_info = copy.deepcopy(trainer.selected_info)
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                # ema_list = copy.deepcopy(trainer.ema_list)
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
                # ema_pool = copy.deepcopy(trainer.ema_pool)
                # task_cossim = copy.deepcopy(trainer.task_cossim)
            
            if "Mutual" in training_args.mode:
                round_info = copy.deepcopy(trainer.round_info)
                selected_info = copy.deepcopy(trainer.selected_info)
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                # ema_list = copy.deepcopy(trainer.ema_list)
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
                # task_mutual = copy.deepcopy(trainer.task_mutual)
            
            if "score" == training_args.mode:
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                # selected_inputs = copy.deepcopy(trainer.selected_inputs)
            
            if training_args.mode in ["divbs", "infoBatch", "gradnorm", "maxloss", "sft"]:
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                # selected_inputs = copy.deepcopy(trainer.selected_inputs)
            
            # if training_args.mode == "sft_org":
                # info_stats = trainer.info_dict
            
            if "Budget" in training_args.mode:
                # np.save(f'embedding_{training_args.note}_round{curr_round}_localrank{training_args.local_rank}.npy', trainer.task_vectors)
                round_info = copy.deepcopy(trainer.round_info)
                selected_info = copy.deepcopy(trainer.selected_info)
                selected_info_ind = copy.deepcopy(trainer.selected_info_ind)
                ema_list = copy.deepcopy(trainer.ema_list)
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
                
                # info_stats = trainer.info_dict
                # np.save(f'{training_args.note}_round{curr_round}.npy', trainer.all_info)
                # np.save(f'embedding_{training_args.note}_round{curr_round}_localrank{training_args.local_rank}.npy', trainer.task_vectors)            
            trainer.accelerator.free_memory()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"done Round {curr_round} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    

        if training_args.resume_train_round > 0 and curr_round+1<training_args.resume_train_round:
            continue
        
        aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **extra_state_dict_dict)
        
        # Save server model
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
        
        elif "sft_org" == training_args.mode:
            comb_info_per_round = {}
            comb_info_per_round["info_stats"] = info_stats
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
        
        
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
        
        if "Budget" in training_args.mode:
            comb_info_per_round = {}
            comb_info_per_round["all_streamed_samples"] = datalist
            comb_info_per_round["all_info_per_round"] = info_per_round
            comb_info_per_round["selected_info_per_round"] = selected_info_per_round
            # comb_info_per_round["selected_inputs"] = selected_inputs
            comb_info_per_round["selected_info_round_per_round"] = selected_info_round_per_round
            comb_info_per_round["selected_input_inds"] = selected_info_ind
            comb_info_per_round["ema_list"] =ema_list
            # comb_info_per_round["info_stats"] = info_stats
            with open(f'importance/{training_args.note}_localrank{training_args.local_rank}_round{curr_round}.json', 'w') as all_info_per_round:
                json.dump(comb_info_per_round, all_info_per_round)
                
        if "GradSim" in training_args.mode:
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
    # if training_args.use_task_vector:
    #     path = os.path.join(training_args.state_dir, f"round{curr_round+1}_task_vector_local_weights.pth")
    #     tv_weight = {'task_vectors': task_vectors, 'local_state_dict_list': local_state_dict_list}
    #     torch.save(tv_weight, path)
        
    #     # task_vector = F.normalize(torch.stack(task_vectors, dim=0), dim=-1)
    #     # sim = torch.matmul(task_vector,
    #     #                 torch.transpose(task_vector, 1, 0))
    #     # sim = torch.transpose(sim, 1, 0)
    #     # sim = (sim+1)/2
        
    #     sims = []
    #     for grad_idx in range(task_vectors[0].shape[-1]):
    #         task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in task_vectors], dim=0), dim=-1)
    #         sim = torch.matmul(task_vector,
    #                         torch.transpose(task_vector, 1, 0))
    #         sim = torch.transpose(sim, 1, 0)
    #         sims.append(sim)
        
    #     sim = torch.stack(sims, dim=0).mean(dim=0)
        
    #     extra_state_dict_dict['task_similarity'] = sim
    #     extra_state_dict_dict['curr_round'] += 1
    logger.info("total done\n")

def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer, processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LLM_LazySupervisedDataset(client_data, tokenizer, data_args, processor)
    data_collator = LLM_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# def make_qwen_supervised_data_module(client_data, tokenizer,
#                                 data_args, model_name, curr_round) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = LLM_LazySupervisedDataset(client_data, tokenizer, data_args, model_id=model_name)
#     data_collator = LLM_DataCollatorForSupervisedDataset(tokenizer=tokenizer.tokenizer)
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=None,
#                 data_collator=data_collator)
# def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer,
#                                 data_args, model_name, curr_round) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""

#     train_dataset = LLM_LazySupervisedDataset(client_data, tokenizer, data_args)
#     data_collator = LLM_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=None,
#                 data_collator=data_collator)

def get_datalists(args, setup, scenario_num):
    with open(f"./scenarios/{setup}/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds
    datasets_name = []
    task_data = scenario[0]
    train_datalist = []
    for task_id, data in enumerate(task_data['datasets']):
            datasets_name.append(data['dataset'])
            with open(f"./datalist/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            # random.shuffle(datalist)
            num_iter = max_iterations #max(int(max_iterations*samplenum_per_rounds/2000), 2) # 10000 / 5 = 2000
            for i in range(rounds_per_task):
                train_datalist.append(
                    {'datalist': datalist,
                     'num_iter': num_iter,
                     'task_id': task_id})
            # with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
            #     datalist = json.load(fp)
            # test_datalist.append({
            #     "data_name": f"{data['dataset']}-{data['subset_id']}",
            #     "data": datalist,
            #     "eval_cnt": eval_cnt})
            # eval_cnt += len(datalist)
            
    return train_datalist, datasets_name

if __name__ == "__main__":
    main()
