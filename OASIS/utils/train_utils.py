import torch
import os
import logging
import transformers
from models.llava.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
import models.llava.conversation as conversation_lib_llava
from peft.tuners.lora import LoraLayer

from models.llava.ema_dual import EMA_LlavaLlamaForCausalLM, EMA_Qwen2_5_VLForConditionalGeneration
from models.internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from models.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from transformers import AutoTokenizer, AutoModel

import copy
from transformers import AutoProcessor, HfArgumentParser
# , Qwen2_5_VLForConditionalGeneration

local_rank = None
def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)
        
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

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["visual","embed_tokens"], verbose=True):
    
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
        
    return lora_module_names

# cls = torch.nn.Linear
#     lora_module_names = set()
#     multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
#     for name, module in model.named_modules():
#         if any(mm_keyword in name for mm_keyword in multimodal_keywords):
#             continue
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     if 'lm_head' in lora_module_names: # needed for 16-bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"
    assert model_args.vision_tower is not None
    if data_args.is_multimodal:
        if 'llava' in model_args.model_name_or_path.lower():
            if model_args.model_type == "mpt":
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    model_max_length=training_args.model_max_length,
                    padding_side="right"
                )
            elif model_args.model_type == 'llama': 
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    model_max_length=training_args.model_max_length,
                    padding_side="right",
                    use_fast=False,
                    token=ACCESS_TOKEN
                )
                
            if tokenizer.unk_token is not None and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
            
            if model_args.model_type == 'llama3-8b':
                tokenizer.pad_token = tokenizer.eos_token
                
            if training_args.is_eval:
                tokenizer.padding_side = "left"
                tokenizer.pad_token = tokenizer.eos_token
            #############################################################################
            if 'cluster' in training_args.mode:
                model = EMA_LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
        
        elif 'qwen' in model_args.model_name_or_path.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            if 'cluster' in training_args.mode:
                model = EMA_Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                # The default setting is padding_side="left"
                # When training using the right-side padding is more efficient.
                    padding_side="right")
            # data_args.image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor
        
        elif 'intern'in model_args.model_name_or_path.lower():
            print("load intern model")
            config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
            config.vision_config.drop_path_rate = 0.01
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            config.template = "internvl2_5"
            config.select_layer = -1
            config.dynamic_image_size = True
            config.use_thumbnail = False
            config.ps_version = 'v2'
            config.min_dynamic_patch = 1
            config.max_dynamic_patch = 6
            model = InternVLChatModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                config=config)
            # model = AutoModel.from_pretrained(model_args.model_name_or_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True).eval().cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False, add_eos_token=False)
            IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            model.img_context_token_id = img_context_token_id
    else:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, token=ACCESS_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            token=ACCESS_TOKEN
        )
        
        
        if tokenizer.unk_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
    
        if training_args.is_eval:
            tokenizer.padding_side = "left"
            
        processor.tokenizer = tokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            use_flash_attention_2=True,
            token=ACCESS_TOKEN
        )
                
        if tokenizer.unk_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        if training_args.is_eval:
            tokenizer.padding_side = "left"
            
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )

    model.config.use_cache = False
    if 'intern' in model_args.model_name_or_path.lower():
        model.requires_grad_(False)
    else:
        model.model.requires_grad_(False)
    # for n, p in model.named_parameters():
    #     print(n)

    # FIXME
    if training_args.bits >= 16:
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
    
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
            
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        # target_modules = ['k_proj', 'v_proj']
        
        if data_args.is_multimodal:
            if 'qwen' in model_args.model_name_or_path.lower():
                lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_target_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
                )
            elif 'intern' in model_args.model_name_or_path.lower():
                if "1b" in model_args.model_name_or_path.lower() or "4b" in model_args.model_name_or_path.lower():
                    target_modules =['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                                'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
                else:
                    target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
                lora_config = LoraConfig(
                r=training_args.lora_r,
                target_modules=target_modules,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                task_type='CAUSAL_LM'
            )
            else:
                lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model), # target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
        else:
            lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            )
        
        if training_args.mode in ['cluster']:
            from models.duallora.dualloramodel import DualLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALLORA'] = DualLoraModel
            lora_config.peft_type = 'DUALLORA'
        
        # model.enable_input_require_grads()
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if data_args.is_multimodal:
        if 'llava' in model_args.model_name_or_path.lower():
            if model_args.version in conversation_lib_llava.conv_templates:
                conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates[model_args.version]
            else:
                conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]
        
        elif 'qwen' in model_args.model_name_or_path.lower():
            if model_args.version in conversation_lib_llava.conv_templates:
                conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates[model_args.version]
            else:
                conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]
    else:
        if 'llama3' in model_args.model_name_or_path.lower() or 'llama-3' in model_args.model_name_or_path.lower():
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates['llama3']
        elif 'qwen3' in model_args.model_name_or_path.lower():
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates['qwen']

    if 'llava' in model_args.model_name_or_path.lower() and data_args.is_multimodal:
        # load vision tower
        # if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            # fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        # if getattr(processor, 'image_processor', None) is not None:
        #     data_args.image_processor = processor.image_processor
        
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = "pad" #data_args.image_aspect_ratio
    
    for n, p in model.named_parameters():
        print(n)
    if 'llava' in model_args.model_name_or_path.lower():
        if "cluster" in training_args.mode:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            
            model.set_state('lora1')
            model.activate_all()
            model.lm_head.requires_grad_(False)
        else:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            model.lm_head.requires_grad_(False)
    else:
        if "cluster" in training_args.mode:
            for p in model.named_parameters():
                if "merger" in p:
                    p.requires_grad = False
            model.set_state('lora1')
            model.activate_all()
            model.lm_head.requires_grad_(False)
        else:
            for p in model.named_parameters():
                if "merger" in p:
                    p.requires_grad = False
            model.lm_head.requires_grad_(False)
    
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        

    if 'qwen' in model_args.model_name_or_path.lower():
            model.config.tokenizer_padding_side = processor.tokenizer.padding_side
            model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    else:    
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    if 'llava' in model_args.model_name_or_path.lower():
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer)or isinstance(module, torch.nn.LayerNorm):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    total_count = 0
    
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)
            total_count += p.numel()
    print(total_count)
    if not data_args.is_multimodal:
        return model, processor, tokenizer, data_args
    if 'qwen' in model_args.model_name_or_path.lower():
        return model, processor, data_args
    else:
        return model, tokenizer, data_args

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

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
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        

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

from torch import nn

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

import random
from selection_methods.fedavg import fedavg_create_trainer

def get_task_vectors(model, tokenizer, datalist, training_args, data_args, global_state_dict, make_supervised_data_module):
    random.seed(training_args.seed)
    client_task_vectors = []

    # for client_id in range(len(train_datalists)):
    #     datalist = train_datalists[client_id][0]['datalist']
        
    # sub_datalist = random.sample(datalist, 4*20)
    data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                            tokenizer=tokenizer,
                                            data_args=copy.deepcopy(data_args))

    extra_state_dict_dict = {}
    extra_state_dict_dict['client_id']=0
    extra_state_dict_dict['curr_round']=0
    extra_state_dict_dict['fisher_freq'] = 1
    # if training_args.selection_method == "info":
    #     training_args.per_gpu_train_batch_size=1
    # else:
    training_args.per_gpu_train_batch_size=4
    training_args.gradient_accumulation_steps=1
    # if training_args.selection_method == "info":
    #     trainer = infoEMA_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, return_task_vectors=True)
    # else:
    trainer = fedavg_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, return_task_vectors=True)

    prev_model_param = trainer.model.state_dict()
    
    output = trainer.train()
    
    # if training_args.selection_method == "info":
    #     task_vectors = trainer.all_info
    # else:
    task_vectors = trainer.task_vector
    print("task_vectors", len(task_vectors))
    
    # client_task_vectors = trainer.cal_task_vector()
    
    trainer.deepspeed.empty_partition_cache()
    del trainer
    
    with torch.no_grad():
        model.load_state_dict(prev_model_param, strict=False)

    # extra_state_dict_dict['fisher_freq']=5
    return task_vectors
