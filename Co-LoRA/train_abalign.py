import logging.config
import os
import random
import gc

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingConfig
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_target_layers

from federated_methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict

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
    
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    
    # select functions
    # set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    
    
    model_ids = {}
    model_list = {}
    models = {}
    processors = {}
    for client_id in range(len(train_datalists)):
        train_datalist = train_datalists[client_id]
        model_id = train_datalist[0]['model_id']
        
        if model_id in model_list.keys():
            continue
        else:
            new_model_args = copy.deepcopy(model_args)
            new_model_args.model_name_or_path = model_id
            model, tokenizer_, processor_, new_data_args = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
            
            models[model_id] = model
            processors[model_id] = (tokenizer_, processor_)
            
            model_ids[model_id] = [client_id]
            
    extra_state_dict_dict = {'model_ids':model_ids}
    
    if training_args.is_cross_model_series:
        from federated_methods.AB_init_cross_model_arch import ABInit_create_trainer
    else:
        from federated_methods.AB_init import ABInit_create_trainer
    
    ##############################################################################################
    # model2_id = "thkim0305/llama3.2_1B_vl" # smaller, anchor model
    model2_id = "thkim0305/qwen2.5_0.5B_vl" # smaller, anchor model
    model_id = 'thkim0305/llama3.2_3B_vl' # larger, training model
    # model keys: thkim0305/llama3.2_3B_vl, thkim0305/llama3.2_1B_vl, thkim0305/llama3.1_8B_vl, thkim0305/qwen2.5_0.5B_vl, thkim0305/qwen2.5_1.5B_vl, thkim0305/qwen2.5_3B_vl
    # llm models: meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct
    model2 = models[model2_id]
    tokenizer2, processor2 = processors[model2_id]
    
    if new_data_args.is_multimodal:
        data_path = 'dataset/llava_finetune/llava_v1_5_mix665k_updated.json'
        public_datalist = json.load(open(data_path, "r"))
        # Filter out items without the "image" key
        public_datalist = [item for item in public_datalist if "image" in item]
    else:
        data_path = 'dataset/chatbotIT.json'
        public_datalist = json.load(open(data_path, "r"))
    random.shuffle(public_datalist)

    # train bigger model
    model = models[model_id]
    tokenizer, processor = processors[model_id]
    model = model.to(torch.bfloat16)
    model2= model2.to(torch.bfloat16)

    ##### A init #####
    state_dict = torch.load('llava_qwen_0_5b_blockwise4_random.pth', map_location='cpu')
    model2.load_state_dict(state_dict, strict=False)

    public_datalist_ = public_datalist[:training_args.AB_align_data_size]
    
    data_module = make_supervised_data_module(client_data=public_datalist_, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args),
                                                model_id=model_id)
    
    trainer = ABInit_create_trainer(model, tokenizer, training_args, data_module, (model2,tokenizer2, processor2, model2_id), data_args, train_A = True)

    results = trainer.train()
    
    output_dir = os.path.join(training_args.state_dir, f"llava_llama_3b_blockwise{training_args.num_blocks}_qwen_A_align.pth")
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), training_args.lora_bias
    )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    state_dict.update(non_lora_state_dict)
    
    torch.save(state_dict, output_dir)
    # ##################################################
    # ensure orthogonality
    # target_layers = [6,13,20,27]
    # target_layers = [7,15,23,31]
    # target_layers = [8,17,26,35]
    # target_layers = [2,5,8,11,14,17,20,27]
    # target_layers = [3,7,11,15,19,23,27,31]
    
    if new_data_args.is_multimodal:
        prefix_ = 'base_model.model.language_model.model.layers'
        target_layers = get_target_layers(len(model.base_model.model.language_model.model.layers), training_args.num_blocks)
    else:
        prefix_ = 'base_model.model.model.layers'
        target_layers = get_target_layers(len(model.base_model.model.model.layers), training_args.num_blocks)
    
    mid_insert = ['self_attn.k_proj','self_attn.q_proj','self_attn.v_proj','self_attn.o_proj', 'mlp.gate_proj','mlp.up_proj','mlp.down_proj']
    for i in target_layers:
        for mid in mid_insert:
            A_key = prefix_ + f'.{i}.' + mid + '.lora_A.default.weight'
            B_key = prefix_ + f'.{i}.' + mid + '.lora_B.default.weight'
            
            A = state_dict[A_key]
            
            A_, err = closest_row_orthonormal(A)
            state_dict[A_key] = A_
    
    torch.save(state_dict, output_dir)
    
    model.load_state_dict(state_dict, strict=False) 
    # ##############################################
    trainer.deepspeed.empty_partition_cache()
    trainer.accelerator.free_memory()
    del trainer
    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # state_dict = torch.load('client_states_debug_qwen_llava_align/llava_3b_orthnormal_init_FT_A.pth',map_location='cpu')
    # model.load_state_dict(state_dict, strict=False) 
    ##### B init #####
    public_datalist_ = random.sample(public_datalist_, 100)
    data_module = make_supervised_data_module(client_data=public_datalist_, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args),
                                                model_id=model_id)
    # model = model.to(torch.bfloat16)
    # model2= model2.to(torch.bfloat16)
    trainer = ABInit_create_trainer(model, tokenizer, training_args, data_module, (model2,tokenizer2, processor2, model2_id), data_args, train_A = False)

    results = trainer.train()
    
    lora_B_output_1b = trainer.lora_B_output_1b
    lora_B_output_3b = trainer.lora_B_output_3b
    
    layer_name_1b = trainer.layer_name_1b
    layer_name_3b = trainer.layer_name_3b
    
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), training_args.lora_bias
    )
    
    state_dict2 = get_peft_state_maybe_zero_3(
        model2.named_parameters(), training_args.lora_bias
    )
    gamma=1e-4
    
    def matrix_inv_sqrt(S):
        S_U, S_S, S_Vt = torch.linalg.svd(S)
        S_neg_sqrt = S_U @ torch.diag(S_S**(-1/2)) @ S_Vt
        S_sqrt = S_U @ torch.diag(S_S**(1/2)) @ S_Vt
        return S_neg_sqrt, S_sqrt
    
    for idx, (X1, X3) in enumerate(zip(lora_B_output_1b, lora_B_output_3b)):
        X1 = X1.cuda()
        X3 = X3.cuda()
        X1 = X1.to(torch.float32)
        X3 = X3.to(torch.float32)
        X1_centered = X1 - X1.mean(dim=0, keepdim=True)
        X3_centered = X3 - X3.mean(dim=0, keepdim=True)
        
        S11 = X1_centered.t() @ X1_centered + gamma*torch.eye(X1.shape[1]).cuda() # shape: [d1, d1]
        S33 = X3_centered.t() @ X3_centered + gamma*torch.eye(X3.shape[1]).cuda() # shape: [d3, d3]
        # S13 = X1_centered.t() @ X3_centered  # shape: [d1, d3]
        S31 = X3_centered.t() @ X1_centered

        S11_neg_sqrt, S11_sqrt = matrix_inv_sqrt(S11.to(torch.float32))
        S33_neg_sqrt, S33_sqrt = matrix_inv_sqrt(S33.to(torch.float32))
        
        # M = S11_neg_sqrt @ S13.to(torch.float32) @ S33_neg_sqrt
        M = S33_neg_sqrt @ S31.to(torch.float32) @ S11_neg_sqrt
        # M will have shape [d1, d3].
        # Perform SVD:  M = U * Sigma * V^T
        U, Sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        
        with torch.no_grad():
            mapping_mat = (S11_neg_sqrt @ Vt.T @ U.T @ S33_sqrt.T).T
            state_dict[layer_name_3b[idx]] = (mapping_mat @ state_dict2[layer_name_1b[idx]].to(torch.float32).cuda()).to(torch.bfloat16).detach().cpu()
            
    for key in state_dict.keys():
        if 'lora_P' in key or 'lora_Q' in key:
            state_dict[key] = torch.zeros_like(state_dict[key])
    
    for key in state_dict2.keys():
        if 'lora_P' in key or 'lora_Q' in key:
            state_dict2[key] = torch.zeros_like(state_dict2[key])
    
    output_dir2 = os.path.join(training_args.state_dir, f'llava_qwen_1b_blockwise{training_args.num_blocks}_random.pth')
    torch.save(state_dict2, output_dir2)
    
    
    output_dir = os.path.join(training_args.state_dir, f'llava_llama_3b_blockwise{training_args.num_blocks}_qwen_AB_align.pth')
    torch.save(state_dict, output_dir)
    return
    ################################################################################################

def closest_row_orthonormal(A: torch.Tensor) -> torch.Tensor:
    """
    Replace A (r × d, r ≤ d) by the closest matrix whose rows are orthonormal.

    Returns
    -------
    A_ortho  # same shape as A, satisfies A_ortho @ A_ortho.T == I_r
    fro_error  # ‖A_ortho − A‖_F  (optional diagnostic)
    """
    orig_dtype = A.dtype
    # SVD in float32 for stability
    U, _, Vt = torch.linalg.svd(A.float(), full_matrices=False)
    A_ortho = (U @ Vt).to(orig_dtype)
    fro_error = torch.norm(A_ortho - A.float(), p='fro').item()
    return A_ortho, fro_error

def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer, processor,
                                data_args, model_id=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args, processor,model_id=model_id)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

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
        for task_id, data in enumerate(client_data['datasets']):
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            num_iter = max_iterations #max(int(max_iterations*samplenum_per_rounds/2000), 2) # 10000 / 5 = 2000
            for i in range(int(rounds_per_task)):
                train_datalist.append(
                    {'datalist':datalist[i*samplenum_per_rounds:(i+1)*samplenum_per_rounds],
                     'num_iter': num_iter,
                     'task_id': task_id,
                     'model_id': client_data['model_id']})
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "type": data['type'],
                "data": datalist,
                "train_start_round": rounds_per_task*task_id})
            
            train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist

    return train_datalists, test_datalists

if __name__ == "__main__":
    main()