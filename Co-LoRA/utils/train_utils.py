import torch
from torch import nn
import numpy as np
import math
import logging
import transformers
import models.llava.conversation as conversation_lib_llava
from peft.tuners.lora import LoraLayer
from models.colora.coloralayer import CoLoraLayer
from models.colora_abinit.colora_init_layer import CoLoraInitLayer
from functools import reduce
import torch.nn.utils as nn_utils

from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

from models.llava.llava_multi import LlavaMultiForConditionalGeneration
from models.llava.llava_fedsim import FEDSIMLlavaMultiForConditionalGeneration
from models.llava.llama_model import CustomLlamaForCausalLM
import copy

from typing import Dict
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset

from datasets import Dataset, load_dataset
import typing as tp
import pandas as pd
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
import random

ACCESS_TOKEN = ""

def deepcopy_layer_with_weight_norm(LayerInit, feature_dim, layer):
    layer_state_dict = layer.state_dict()

    new_layer = LayerInit(feature_dim)

    new_layer.load_state_dict(layer_state_dict)

    return new_layer


def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"

    # load tokenizer
    # for llava
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    if training_args.is_eval:
        tokenizer.padding_side = "left"
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
    if data_args.is_multimodal:
        if 'llava' in model_args.model_name_or_path.lower() or 'vl' in model_args.model_name_or_path.lower():
            if 'fedsim' in training_args.mode:
                model = FEDSIMLlavaMultiForConditionalGeneration.from_pretrained( # LlavaForConditionalGeneration
                    model_args.model_name_or_path,
                    torch_dtype=compute_dtype,
                    use_flash_attention_2=True,
                    token=ACCESS_TOKEN
                )
            else:
                model = LlavaMultiForConditionalGeneration.from_pretrained( # LlavaForConditionalGeneration
                    model_args.model_name_or_path,
                    torch_dtype=compute_dtype,
                    use_flash_attention_2=True,
                    quantization_config = bnb_model_from_pretrained_args if bnb_model_from_pretrained_args else None,
                    token=ACCESS_TOKEN
                )
        else:
            model = AutoModelForImageTextToText.from_pretrained( # LlavaForConditionalGeneration
                model_args.model_name_or_path,
                torch_dtype=compute_dtype,
                use_flash_attention_2=True,
                token=ACCESS_TOKEN
            )
    else:
        if 'llama' in model_args.model_name_or_path.lower():
            model = CustomLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=compute_dtype,
                use_flash_attention_2=True,
                token=ACCESS_TOKEN
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=compute_dtype,
                use_flash_attention_2=True,
                token=ACCESS_TOKEN
            )
    model.config.use_cache = False
    if getattr(model, 'vision_tower', None) is not None:
        model.vision_tower.requires_grad_(False)
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )
    if bnb_model_from_pretrained_args is None:
        model = model.to(training_args.device)
    
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if bnb_model_from_pretrained_args is None and training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            exclude_modules=r".*vision_tower.*|.*multi_modal_projector.*", 
        )
        
        if training_args.mode in ['perada','fedsim','ditto','feddpa']:
            from models.duallora.dualloramodel import DualLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALLORA'] = DualLoraModel
            lora_config.peft_type = 'DUALLORA'
        
        elif training_args.mode in ['feddat']:
            from models.triplelora.tripleloramodel import TripleLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['TRILORA'] = TripleLoraModel
            lora_config.peft_type = 'TRILORA'
        
        elif training_args.mode in ['fedmosaic_homo']:
            from models.duallora_moe.dualmoeloramodel import DualMOELoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALMOELORA'] = DualMOELoraModel
            lora_config.peft_type = 'DUALMOELORA'

        elif training_args.mode in ['fedmosaic']:
            from models.colora.coloramodel import CoLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['COLORA'] = CoLoraModel
            lora_config.peft_type = 'COLORA'
        
        elif training_args.mode in ['colora_ABinit']:
            from models.colora_abinit.colora_init_model import CoLoraInitModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['COLORAINIT'] = CoLoraInitModel
            lora_config.peft_type = 'COLORAINIT'
            
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'llama3' in model_args.model_name_or_path.lower() or 'llama-3' in model_args.model_name_or_path.lower():
        conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates['llama3']
    elif 'qwen2' in model_args.model_name_or_path.lower():
        conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates['qwen']
    else:
        conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]

    if getattr(processor, 'image_processor', None) is not None:
        data_args.image_processor = processor.image_processor

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # freeze some layers
    if data_args.is_multimodal:
        total_layers = model.base_model.language_model.model.layers
    else:
        if training_args.is_t5model:
            total_layers = model.base_model.model.encoder.block + model.base_model.model.decoder.block
        else:
            total_layers = model.base_model.model.model.layers
    
    #FIXME
    # lora initialization & freeze
    if training_args.mode == 'colora_ABinit':
        target_layers = get_target_layers(len(total_layers), training_args.num_blocks)
            
        for idx, layer in enumerate(total_layers):
            if idx in target_layers:
                for n, m in layer.named_modules():
                    if 'lora_A.default' in n or 'lora_B.default' in n:
                        m.apply(orthonormal_kaiming_uniform_init)
                        # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                        m.weight.to(torch.bfloat16)
                for n, p in layer.named_parameters():
                    if 'lora_A' in n:
                        p.requires_grad = True
                    elif 'lora_B' in n:
                        p.requires_grad = False
                    elif 'lora_P' in n or 'lora_Q' in n:
                        nn.init.zeros_(p)
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, CoLoraInitLayer):
                        m.use_pq = False
                for n, p in layer.named_parameters():
                    p.requires_grad = False
    
    elif training_args.mode in ['fedmosaic']:
        target_layers = get_target_layers(len(total_layers), training_args.num_blocks)
        for idx, layer in enumerate(total_layers):
            if idx in target_layers:
                for n, p in layer.named_parameters():
                    if 'lora1_A' in n or 'lora2_A' in n:
                        p.requires_grad = False
                    elif 'lora1_B' in n:
                        p.requires_grad = False
                        init_B = torch.empty_like(p)
                        nn.init.kaiming_uniform_(init_B, a=math.sqrt(5))
                        # Get the current shape of the weight matrix
                        rows, cols = p.size()

                        # Ensure the matrix is contiguous
                        init_B = init_B.contiguous()

                        # Perform Singular Value Decomposition
                        u, _, v = torch.svd(init_B, some=False)
                        
                        u = u.contiguous()
                        v = v.contiguous()

                        # Use U or V from SVD based on the shape of the weight matrix
                        if rows > cols:
                            init_B.data = u[:, :cols].to(torch.bfloat16)
                        else:
                            init_B.data = v[:rows, :].to(torch.bfloat16)
                        p.data = copy.deepcopy(init_B)
                        new_n = n.replace('lora1_B', 'lora2_B')
                        dict(layer.named_parameters())[new_n].data = copy.deepcopy(init_B)
                        dict(layer.named_parameters())[new_n].requires_grad = False
                    elif 'lora1_P' in n or 'lora2_P' in n or 'lora1_Q' in n or 'lora2_Q' in n:
                        nn.init.zeros_(p)
                for n, m in layer.named_modules():
                    if isinstance(m, CoLoraLayer):
                        m.freeze_AB = True
            else:
                for n, m in layer.named_modules():
                    if isinstance(m, CoLoraLayer):
                        m.use_pq = False
              
    model.config.mm_projector_lr = training_args.mm_projector_lr
    
    if hasattr(model.config, "image_token_index"):
        tokenizer.image_token_index = model.config.image_token_index
    
    # load pretrained lora
    if training_args.load_pretrained_lora:
        if training_args.mode in ['fedmosaic']:
            if not data_args.is_multimodal:
                if 'Llama-3.2-1B' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llama_1b_blockwise{training_args.num_blocks}_random.pth', map_location='cpu')
                elif 'Llama-3.2-3B' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llama_3b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
                elif 'Llama-3.1-8B' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llama_8b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
            elif training_args.is_cross_model_series:
                if 'qwen2.5_0.5B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llava_qwen_0_5b_blockwise{training_args.num_blocks}_random.pth', map_location='cpu')
                elif 'qwen2.5_1.5B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llava_qwen_1_5b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
                elif 'qwen2.5_3B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llava_qwen_3b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
                elif 'llama3.2_1B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llava_llama_1b_blockwise{training_args.num_blocks}_qwen_AB_align.pth', map_location='cpu')
                elif 'llama3.2_3B_vl' in model_args.model_name_or_path:
                    state_dict = torch.load(f'llava_llama_3b_blockwise{training_args.num_blocks}_qwen_AB_align.pth', map_location='cpu')
                
            elif 'llama3.2_1B_vl' in model_args.model_name_or_path:
                state_dict = torch.load(f'llava_llama_1b_blockwise{training_args.num_blocks}_random.pth', map_location='cpu')
            elif 'llama3.2_3B_vl' in model_args.model_name_or_path:
                state_dict = torch.load(f'llava_llama_3b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
            elif 'llama3.1_8B_vl' in model_args.model_name_or_path:
                state_dict = torch.load(f'llava_llama_8b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
            elif 'qwen2.5_0.5B_vl' in model_args.model_name_or_path:
                state_dict = torch.load(f'llava_qwen_0_5b_blockwise{training_args.num_blocks}_random.pth', map_location='cpu')
            elif 'qwen2.5_1.5B_vl' in model_args.model_name_or_path:
                state_dict = torch.load(f'llava_qwen_1_5b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
            elif 'qwen2.5_3B_vl' in model_args.model_name_or_path:
                state_dict = torch.load(f'llava_qwen_3b_blockwise{training_args.num_blocks}_AB_align.pth', map_location='cpu')
            
            
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k1 = k.replace('lora', 'lora1')
                new_k2 = k.replace('lora', 'lora2')
                
                new_state_dict[new_k1] = v
                new_state_dict[new_k2] = v

            model.load_state_dict(new_state_dict, strict=False)

        print('load pretrained lora')
    
    total_count = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            # print(n, p.shape)
            p.data = p.data.to(compute_dtype) 
            total_count += p.numel()
    print(f"trainable param num: {total_count}")
    return model, tokenizer, processor, data_args

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'mlp1', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters



# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

    
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def load_deepspeed(state_dict, module: nn.Module, prefix="", strict=True):
    import deepspeed
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix, {}, strict, [], [], [])
            # module.load_state_dict(state_dict, strict=strict)

    for name, child in module._modules.items():
        if child is not None:
            load_deepspeed(state_dict, child, prefix + name + ".")


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def configure_online_datastream(sub_dataset, num_iterations, training_args, client_id, memory, memory_count, memory_size, total_batchsize):
    iteration = 0
    datalist = []
    iter_ratio = num_iterations / len(sub_dataset)
    T = training_args.online_stream_T
    count_decay_ratio = training_args.online_stream_count_decay_ratio
    
    if not training_args.is_streamonly:
        # memory only: priority-based sampling
        for i, sample in enumerate(sub_dataset):
            if len(memory[client_id]) == memory_size:
                pop_index = random.randrange(memory_size)
                memory[client_id].pop(pop_index)
                memory_count[client_id] = np.delete(memory_count[client_id], pop_index, 0)
            
            memory[client_id].append(sample)
            memory_count[client_id] = np.append(memory_count[client_id], 0)
            iteration += iter_ratio
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
    else:
        # stream-only
        datalist = sub_dataset[:num_iterations*total_batchsize]
    
    return datalist

def get_keys_to_del(training_args, new_global_state_dict, data_args):
    keys_to_del = []
    layer_index = 5 if data_args.is_multimodal else 4
    if training_args.mode in ['fedmosaic_homo', 'fedmosaic']:
        for k in new_global_state_dict.keys():
            if 'lora2' in k or 'lora_w_gate' in k or 'lora_w_noise' in k:
                keys_to_del.append(k)

    return keys_to_del

def get_target_layers(total_num_layers, num_blocks=8):
    step = total_num_layers // num_blocks
    target_layers = [
        k * step - 1 for k in range(1, num_blocks)
    ] + [total_num_layers - 1]
    return target_layers

def orthonormal_kaiming_uniform_init(m):
    if isinstance(m, nn.Linear):
        # Get the current shape of the weight matrix
        rows, cols = m.weight.size()

        # Initialize the weight matrix with Kaiming Uniform
        a = nn.init.kaiming_uniform_(torch.empty(rows, cols).cuda(), a=math.sqrt(5))

        # Ensure the matrix is contiguous
        a = a.contiguous()

        # Perform Singular Value Decomposition
        u, _, v = torch.svd(a, some=False)
        
        u = u.contiguous()
        v = v.contiguous()

        # Use U or V from SVD based on the shape of the weight matrix
        if rows > cols:
            m.weight.data = u[:, :cols].to(torch.bfloat16)
        else:
            m.weight.data = v[:rows, :].to(torch.bfloat16)

        # Optional: Initialize biases to zero
        if m.bias is not None:
            m.bias.data.zero_()


def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer, processor,
                                data_args, model_id=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args, processor, model_id=model_id)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
