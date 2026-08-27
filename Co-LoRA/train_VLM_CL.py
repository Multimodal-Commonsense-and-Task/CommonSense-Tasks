import logging.config
import os
import random
import gc

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingConfig
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, configure_online_datastream, get_keys_to_del, make_supervised_data_module

from federated_methods.method_manager import select_method

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

os.environ["WANDB_DISABLED"] = "true"
def main():    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = None
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
    if 'feddat' in training_args.mode or 'perada' in training_args.mode or 'ditto' in training_args.mode:
        training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps//2
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

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    
    # select functions
    set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    
    LAYER_INDEX = 5 if data_args.is_multimodal else 4
    
    model_ids = {}
    model_list = {}
    models = {}
    processors = {}
    global_state_dict_list = []
    local_state_dict_list = []
    old_local_state_dict_list = []
    for client_id in range(len(train_datalists)):
        train_datalist = train_datalists[client_id]
        model_id = train_datalist[0]['model_id']
        
        if model_id in model_list.keys():
            local_state_dict_list.append(copy.deepcopy(model_list[model_id]))
            old_local_state_dict_list.append(copy.deepcopy(model_list[model_id]))
            global_state_dict = copy.deepcopy(model_list[model_id])
            keys_to_del = get_keys_to_del(training_args, global_state_dict, data_args)
            for k in keys_to_del:
                del global_state_dict[k]
            global_state_dict_list.append(global_state_dict)
            
            model_ids[model_id].append(client_id)
        else:
            new_model_args = copy.deepcopy(model_args)
            new_model_args.model_name_or_path = model_id
            model, tokenizer_, processor_, new_data_args = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
            
            global_state_dict = get_peft_state_maybe_zero_3(
                        model.named_parameters(), training_args.lora_bias
                    )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            global_state_dict.update(non_lora_state_dict)
            
            local_state_dict_list.append(copy.deepcopy(global_state_dict))
            old_local_state_dict_list.append(copy.deepcopy(global_state_dict))
            new_global_state_dict=copy.deepcopy(global_state_dict)
            keys_to_del = get_keys_to_del(training_args, new_global_state_dict, data_args)
            for k in keys_to_del:
                del new_global_state_dict[k]
            global_state_dict_list.append(new_global_state_dict)
            
            model_list[model_id] = global_state_dict
            
            model = model.cpu()
            models[model_id] = model
            processors[model_id] = (tokenizer_, processor_)
            model_ids[model_id] = [client_id]
    
    if data_args.is_multimodal and training_args.is_cross_model_series and 'thkim0305/qwen2.5_0.5B_vl' not in models.keys():
        new_model_args = copy.deepcopy(model_args)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit       = True,      # or  load_in_8bit=True
            bnb_4bit_quant_type= "nf4",     # "fp4" or "nf4"; ignore for 8‑bit
            bnb_4bit_compute_dtype = torch.bfloat16,  # compute in bf16 (or fp16)
            bnb_4bit_use_double_quant = True,         # slightly slower / less RAM
        )
        new_model_args.model_name_or_path = 'thkim0305/qwen2.5_0.5B_vl'
        model2, tokenizer_,processor_,_ = get_VLMmodel(new_model_args, training_args, bnb_config, data_args)
        models['thkim0305/qwen2.5_0.5B_vl'] = model2
        processors['thkim0305/qwen2.5_0.5B_vl'] = (tokenizer_, processor_)
    elif data_args.is_multimodal and training_args.use_task_vector and 'llama3.' in model_id and 'thkim0305/llama3.2_1B_vl' not in models.keys():
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = 'thkim0305/llama3.2_1B_vl'
        model2, tokenizer_,processor_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
        models['thkim0305/llama3.2_1B_vl'] = model2
        processors['rthkim0305/llama3.2_1B_vl'] = (tokenizer_, processor_)
    elif data_args.is_multimodal and training_args.use_task_vector and 'qwen2.5' in model_id and 'thkim0305/qwen2.5_0.5B_vl' not in models.keys():
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = 'thkim0305/qwen2.5_0.5B_vl'
        model2, tokenizer_,processor_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
        models['thkim0305/qwen2.5_0.5B_vl'] = model2
        processors['thkim0305/qwen2.5_0.5B_vl'] = (tokenizer_, processor_)
    elif not data_args.is_multimodal and training_args.use_task_vector and 'meta-llama/Llama-3.2-1B' not in models.keys():
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = 'meta-llama/Llama-3.2-1B'
        model2, tokenizer_,processor_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
        models['meta-llama/Llama-3.2-1B'] = model2
        processors['meta-llama/Llama-3.2-1B'] = (tokenizer_, processor_)
    
    del model_list
    extra_state_dict_dict = {'model_ids':model_ids, 'models':models, 'processors':processors}
    extra_state_dict_dict['LAYER_INDEX'] = LAYER_INDEX
    
    torch.cuda.empty_cache()

    training_loss = [[] for i in range(training_args.num_clients)]
    
    # start federated learning
    start_time = time.time()
    frac_clients = 1
    
    memory = [[] for id in range(training_args.num_clients)]
    memory_count = [np.array([]) for id in range(training_args.num_clients)]
    memory_size = training_args.memory_size
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    
    total_rounds = int(training_args.num_rounds * training_args.num_tasks)
    last_task_id = [-1 for _ in range(training_args.num_clients)]
    grad_olds = [None for _ in range(training_args.num_clients)]
    task_vectors = [None for _ in range(training_args.num_clients)]
    
    original_weights = {}
    lr_step = (init_lr - final_lr)/total_rounds
    mm_lr_step = (mm_init_lr - mm_final_lr)/total_rounds
    for curr_round in range(total_rounds):
        old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
        
        if curr_round > 0 and training_args.use_task_vector:
            path = os.path.join(training_args.state_dir, f"round{curr_round}_task_vector_local_weights.pth")
            tv_weight = {'task_vectors': task_vectors}
            torch.save(tv_weight, path)
            
            # task vector layerwise cosine sim
            sims = []
            for grad_idx in range(task_vectors[0].shape[-1]):
                task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in task_vectors[:]], dim=0), dim=-1)
                sim = torch.matmul(task_vector,
                                torch.transpose(task_vector, 1, 0))
                sim = torch.transpose(sim, 1, 0)
                sims.append(sim)
                
            sim = torch.stack(sims, dim=0).mean(dim=0)
            # sim = torch.ones(10,10)
            
            extra_state_dict_dict['task_similarity'] = sim
            print("task similarity matrix:")
            print(sim)
        
        # clients turn
        cids = np.arange(training_args.num_clients).tolist()
        num_selection = int(round(training_args.num_clients*frac_clients)) 
        selected_ids = sorted(random.sample(cids, num_selection)) 
        
        if training_args.local_rank == 0 or training_args.local_rank == -1: 
            logger.info(f"Round {curr_round} | selected_ids: {selected_ids}\n")
        
        extra_state_dict_dict['selected_ids'] = selected_ids
        extra_state_dict_dict['num_selection'] = num_selection
        
        # selected_ids = cids
        training_args.learning_rate = init_lr - lr_step*curr_round
        training_args.mm_projector_lr = mm_init_lr - mm_lr_step*curr_round
        for idx in range(num_selection):
            client_id = selected_ids[idx]
            
            ##### simulate online memory insertion & get_batch ####
            sub_dataset = train_datalists[client_id][curr_round]['datalist']
            num_iterations = train_datalists[client_id][curr_round]['num_iter']
            
            task_id = train_datalists[client_id][curr_round]['task_id']
            
            test_datalist = test_datalists[client_id]
            
            model_id = train_datalists[client_id][curr_round]['model_id']
            new_model_args = copy.deepcopy(model_args)
            new_model_args.model_name_or_path = model_id
            new_data_args = copy.deepcopy(data_args)
            new_data_args.model_name_for_dataarg = model_id
            # model,_,_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, new_data_args)
            model = models[model_id]
            tokenizer, processor = processors[model_id]
            
            extra_state_dict_dict['client_id'] = client_id
            extra_state_dict_dict['curr_round'] = curr_round
            extra_state_dict_dict['test_datalist'] = test_datalist
            extra_state_dict_dict['processor'] = processor
            extra_state_dict_dict['data_args'] = copy.deepcopy(new_data_args)
            extra_state_dict_dict['tokenizer'] = tokenizer
            
            new_global_state_dict = None
            new_global_state_dict = load_state_dict(model, global_state_dict_list[client_id], old_local_state_dict_list, client_id, training_args, extra_state_dict_dict)
            print('model loading done')
            
            if new_global_state_dict is not None:
                extra_state_dict_dict['new_global_state_dict'] = new_global_state_dict
            
            datalist = configure_online_datastream(sub_dataset, num_iterations, training_args, client_id, memory, memory_count, memory_size, total_batchsize)
            data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args),
                                                model_id=model_id)
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train client {client_id} | num samples {len(sub_dataset)}')

            # ===== Train local model on the client side =====
            if training_args.use_task_vector:
                extra_state_dict_dict['grad_old'] = grad_olds[client_id]
                extra_state_dict_dict['task_vector'] = task_vectors[client_id]
                extra_state_dict_dict['grad_freq'] = training_args.grad_freq
                if data_args.is_multimodal:
                    if 'thkim0305/qwen2.5_0.5B_vl' in models.keys():
                        # extra_state_dict_dict['model2'] = models['thkim0305/qwen2.5_0.5B_vl']
                        tokenizer2, processor2 = processors['thkim0305/qwen2.5_0.5B_vl']
                        extra_state_dict_dict['model2'] = (models['thkim0305/qwen2.5_0.5B_vl'], tokenizer2, processor2, 'thkim0305/qwen2.5_0.5B_vl')
                    elif 'thkim0305/llama3.2_1B_vl' in models.keys():
                        # extra_state_dict_dict['model2'] = models['thkim0305/llama3.2_1B_vl']
                        tokenizer2, processor2 = processors['thkim0305/llama3.2_1B_vl']
                        extra_state_dict_dict['model2'] = (models['thkim0305/llama3.2_1B_vl'], tokenizer2, processor2, 'thkim0305/llama3.2_1B_vl')
                else:
                    # extra_state_dict_dict['model2'] = models['meta-llama/Llama-3.2-1B']
                    tokenizer2, processor2 = processors['meta-llama/Llama-3.2-1B']
                    extra_state_dict_dict['model2'] = (models['meta-llama/Llama-3.2-1B'], tokenizer2, processor2, 'meta-llama/Llama-3.2-1B')
            
            trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)

            results = trainer.train()
            training_loss[client_id].append(results.training_loss)
                     
            if training_args.use_task_vector:
                grad_olds[client_id] = trainer.grad_old
                task_vectors[client_id] = trainer.task_vector
                
                if data_args.is_multimodal:
                    if 'thkim0305/qwen2.5_0.5B_vl' in models.keys():
                        extra_state_dict_dict['model2'] = models['thkim0305/qwen2.5_0.5B_vl'].cpu()
                    elif 'thkim0305/llama3.2_1B_vl' in models.keys():
                        extra_state_dict_dict['model2'] = models['thkim0305/llama3.2_1B_vl'].cpu()
                else:
                    models['meta-llama/Llama-3.2-1B'] = models['meta-llama/Llama-3.2-1B'].cpu()
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                path = os.path.join(training_args.state_dir, f"{client_id}_trainer_state.json")
                trainer.state.save_to_json(path)
            
            model.config.use_cache = True
            
            # save local model
            output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")

            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            state_dict.update(non_lora_state_dict)
            
            # if 'full' in training_args.mode:
            #     print("Determinant of P")
            #     for k in state_dict.keys():
            #         if 'lora_P' in k or 'lora1_P' in k or 'lora2_P' in k:
            #             square_matrix = state_dict[k].to(torch.float32)
            #             print(k, torch.det(square_matrix))
            
            if (training_args.local_rank == 0 or training_args.local_rank == -1):# and (curr_round+1)%(5) == 0:
                torch.save(state_dict, output_dir)
            
            local_state_dict_list[client_id] = copy.deepcopy(state_dict)
            
            # local_state_dict = getattr(trainer, 'global_weight', None)
            # if local_state_dict is not None:
            #     local_state_dict_list[client_id] = copy.deepcopy(local_state_dict)
            
            trainer.deepspeed.empty_partition_cache()
            trainer.accelerator.free_memory()
            del trainer
            model = model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"done Round {curr_round} client {client_id} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
            
        aggregate_state_dict(global_state_dict_list, local_state_dict_list, training_args=training_args, **extra_state_dict_dict)
        
    if training_args.use_task_vector:
        path = os.path.join(training_args.state_dir, f"round{curr_round+1}_task_vector_local_weights.pth")
        tv_weight = {'task_vectors': task_vectors}#, 'local_state_dict_list': local_state_dict_list}
        torch.save(tv_weight, path)
    logger.info("total done\n")

def get_datalists(args, scenario_num):
    with open(f"./scenarios/{scenario_num}.json") as fp:
        scenario = json.load(fp)
    
    assert args.num_clients == len(scenario)

    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        
        if args.is_continual: # PFCL
            # Calculate total number of rounds
            total_rounds = int(rounds_per_task * len(client_data['datasets']))
            
            # Load all task datasets first
            task_datasets = []
            for task_id, data in enumerate(client_data['datasets']):
                if data['dataset'] == 'dummy':
                    task_datasets.append({
                        'task_id': task_id,
                        'is_dummy': True,
                        'data': [],
                        'dataset_info': data
                    })
                else:
                    with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                        datalist = json.load(fp)
                    random.shuffle(datalist)
                    task_datasets.append({
                        'task_id': task_id,
                        'is_dummy': False,
                        'data': datalist,
                        'dataset_info': data
                    })
            
            # Create rounds by distributing task data proportionally
            for round_idx in range(total_rounds):
                # Calculate which task(s) this round spans
                round_start = round_idx / rounds_per_task  # fractional task position
                round_end = (round_idx + 1) / rounds_per_task
                
                round_data = []
                primary_task_id = int(round_start)  # The main task for this round
                
                # Handle case where round spans multiple tasks
                start_task = int(round_start)
                end_task = int(round_end)
                
                if start_task == end_task:
                    # Round is entirely within one task
                    task_info = task_datasets[start_task]
                    if not task_info['is_dummy']:
                        task_data = task_info['data']
                        # Calculate which portion of this task belongs to this round
                        task_progress_start = round_start - start_task
                        task_progress_end = round_end - start_task
                        # print(task_progress_start, task_progress_end)
                        start_idx = int(task_progress_start * len(task_data))
                        end_idx = int(task_progress_end * len(task_data))
                        round_data = task_data[start_idx:end_idx]
                else:
                    # Round spans multiple tasks
                    for task_idx in range(start_task, end_task + 1):
                        if task_idx >= len(task_datasets):
                            break
                            
                        task_info = task_datasets[task_idx]
                        if task_info['is_dummy']:
                            continue
                            
                        task_data = task_info['data']
                        
                        if task_idx == start_task:
                            # First task: from round_start to end of task
                            task_progress_start = round_start - task_idx
                            start_idx = int(task_progress_start * len(task_data))
                            round_data.extend(task_data[start_idx:])
                        elif task_idx == end_task:
                            # Last task: from beginning to round_end
                            task_progress_end = round_end - task_idx
                            end_idx = int(task_progress_end * len(task_data))
                            round_data.extend(task_data[:end_idx])
                        else:
                            # Middle task: entire task
                            round_data.extend(task_data)
                # Ensure we have a valid primary task ID
                primary_task_id = min(primary_task_id, len(client_data['datasets']) - 1)
                
                num_iter = max_iterations
                train_datalist.append({
                    'datalist': round_data,
                    'num_iter': num_iter,
                    'task_id': primary_task_id,
                    'model_id': client_data['model_id']
                })
            
            # Create test datalists (unchanged logic for test data)
            for task_id, data in enumerate(client_data['datasets']):
                if data['dataset'] == 'dummy':
                    test_datalist.append({'data': [], 'type': ''})
                    continue
                    
                with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                    datalist = json.load(fp)
                
                test_datalist.append({
                    "data_name": f"{data['dataset']}-{data['subset_id']}",
                    "type": data['type'],
                    "data": datalist,
                    "train_start_round": int(rounds_per_task * task_id)
                })
            
            train_datalists[client_id] = train_datalist
            test_datalists[client_id] = test_datalist
        else: # PFL
            combined_datalist = []
            for task_id, data in enumerate(client_data['datasets']):
                with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                    datalist = json.load(fp)
                combined_datalist.extend(datalist)
            random.shuffle(combined_datalist)
            samplenum_per_rounds = int(len(combined_datalist)/ (args.num_rounds * args.num_tasks))
            num_iter = max_iterations
            for i in range(args.num_rounds * args.num_tasks):
                train_datalist.append(
                    {'datalist':combined_datalist[i*samplenum_per_rounds:(i+1)*samplenum_per_rounds],
                    'num_iter': num_iter,
                    'task_id': 0,
                    'model_id': client_data['model_id']})
            train_datalists[client_id] = train_datalist
            test_datalists[client_id] = test_datalist

    return train_datalists, test_datalists

if __name__ == "__main__":
    main()
