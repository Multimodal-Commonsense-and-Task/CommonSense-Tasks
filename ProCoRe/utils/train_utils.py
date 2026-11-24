import torch
import os
import logging
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
import models.llava.conversation as conversation_lib_llava
# from peft.tuners.lora import LoraLayer
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor, AutoModelForImageTextToText
import math
import copy
from transformers import HfArgumentParser

# , Qwen2_5_VLForConditionalGeneration

ACCESS_TOKEN = os.environ.get("HF_TOKEN")  # set HF_TOKEN in your shell

local_rank = None
def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def split_model(model_name, model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

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


def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = training_args.attn_impl
    print("attn_implementation", attn_implementation )

    use_dara = False
    use_only = False
    if training_args.mode == "dara":
        use_dara = True
        print("use_dara")
    if training_args.mode == "only":
        use_only = True
        print("use_only")
    assert model_args.vision_tower is not None
    
    if training_args.use_vis_token_amplifier == True:
        use_vis_token_amplifier = True
        use_pai = False
    elif training_args.use_pai == True:
        use_pai = True
        use_vis_token_amplifier = False
    else:
        use_vis_token_amplifier = False
        use_pai = False
    if training_args.use_icot:
        use_icot = True
    else:
        use_icot = False

    # load tokenizer
    # for llava
    if 'llava' in model_args.model_name_or_path.lower():
        if model_args.model_type == "mpt":
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right"
            )
    
        elif model_args.model_type == 'llama': 
            # if "13" in model_args.model_name_or_path:
            #     tokenizer = transformers.AutoTokenizer.from_pretrained(
            #         model_args.model_name_or_path,
            #         cache_dir="/home/mila/s/sparsha.mishra/scratch/lmj",
            #         model_max_length=training_args.model_max_length,
            #         padding_side="right",
            #         use_fast=False,
            #     )
            # else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
                )

    if 'idefics'in model_args.model_name_or_path.lower():
        from models.idefics3.modeling_idefics3 import My_Idefics3ForConditionalGeneration
        from models.idefics3.modeling_idefics3_mimic import My_Idefics3ForConditionalGeneration_mimic
        from models.idefics3.modeling_idefics3_dara import My_Idefics3ForConditionalGeneration_dara
        if use_dara:
            model = My_Idefics3ForConditionalGeneration_dara.from_pretrained(
                model_args.model_name_or_path, torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)), device_map=("auto" if os.environ.get("P40_DEVICE_MAP") == "auto" else "cuda"), low_cpu_mem_usage=True, use_dara=use_dara,max_images=int(training_args.train_scenario)+1
            )
        elif training_args.mode == "mimic":
            model = My_Idefics3ForConditionalGeneration_mimic.from_pretrained(
                model_args.model_name_or_path, torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)), device_map=("auto" if os.environ.get("P40_DEVICE_MAP") == "auto" else "cuda"), low_cpu_mem_usage=True, use_dara=use_dara,max_images=int(training_args.train_scenario)+1
            )
            model.model.text_model.add_mimic_module(
                enable_attn_shift=True,
                attn_shift_multihead=True,
                learnable_mu=False,
                keep_logZ1_fp32=True,
            )
        else:
            model = My_Idefics3ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path, torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)), device_map=("auto" if os.environ.get("P40_DEVICE_MAP") == "auto" else "cuda"), low_cpu_mem_usage=True, vis_token_amplifier=training_args.vis_token_amplifier,attn_implementation=attn_implementation
            )
            
        if training_args.mode=="dara":
            model.model.text_model.add_dara_module(
                max_images=int(training_args.train_scenario)+1,           
                init_value=1.0,
            )
        
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        tokenizer = processor.tokenizer
    elif 'onevision-qwen'in model_args.model_name_or_path.lower():
        from models.llava.language_model.llava_qwen import LlavaQwenForCausalLM
        from transformers import LlavaOnevisionForConditionalGeneration
        # if use_dara
        model = LlavaQwenForCausalLM.from_pretrained(model_args.model_name_or_path,  cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
            ignore_mismatched_sizes=True,
            use_dara=use_dara,
            max_images=int(training_args.train_scenario)+1)
        # model = LlavaQwenForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        #     attn_implementation=attn_implementation,
        #     torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
        #     ignore_mismatched_sizes=True
        #     )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        data_args.image_processor = processor.image_processor
    elif 'onevision-1.5'in model_args.model_name_or_path.lower():
        from models.llavaonevision1_5.modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration
        from models.llavaonevision1_5.modeling_llavaonevision1_5_mimic import LLaVAOneVision1_5_ForConditionalGeneration_mimic
        
        if use_vis_token_amplifier:
            model = LLaVAOneVision1_5_ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path, attn_implementation=attn_implementation, torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)), device_map=("auto" if os.environ.get("P40_DEVICE_MAP") == "auto" else "cuda"), trust_remote_code=True, vis_token_amplifier=training_args.vis_token_amplifier
            )
        elif training_args.mode == "mimic":
            model = LLaVAOneVision1_5_ForConditionalGeneration_mimic.from_pretrained(
                model_args.model_name_or_path, attn_implementation=attn_implementation, torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)), device_map=("auto" if os.environ.get("P40_DEVICE_MAP") == "auto" else "cuda"), trust_remote_code=True, vis_token_amplifier=training_args.vis_token_amplifier
            )
            model.model.language_model.add_mimic_module(
                enable_attn_shift=True,
                attn_shift_multihead=True,
                learnable_mu=True,
                keep_logZ1_fp32=True,
            )
        else:
            model = LLaVAOneVision1_5_ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path, attn_implementation=attn_implementation, torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)), device_map=("auto" if os.environ.get("P40_DEVICE_MAP") == "auto" else "cuda"), trust_remote_code=True, use_dara=use_dara, max_images=int(training_args.train_scenario)+1,
            )
        
        if training_args.mode == "dara":
            model.model.language_model.add_dara_module(
                max_images=int(training_args.train_scenario)+1,           
                init_value=1.0,
            )
        
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        data_args.image_processor = processor.image_processor
    elif 'llava' in model_args.model_name_or_path.lower():
        from models.llava.language_model.llava_llama import LlavaLlamaForCausalLM
        if tokenizer.unk_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        
        if model_args.model_type == 'llama3-8b':
            tokenizer.pad_token = tokenizer.eos_token
            
        if training_args.is_eval:
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        #############################################################################
    
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
            ignore_mismatched_sizes=True,
            **bnb_model_from_pretrained_args
        )
    
    
    elif 'internvl3_5' in model_args.model_name_or_path.lower():
        from models.internvl3_5.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
        from models.internvl3_5.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
        from models.internvl3_5.model.internvl_chat.modeling_internvl_chat_mimic import InternVLChatModel_mimic
        
        if training_args.mode == "mimic":
            model = InternVLChatModel_mimic.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                use_flash_attn=False,
                trust_remote_code=True,
                **({"device_map": split_model(None, model_args.model_name_or_path)} if os.environ.get("P40_DEVICE_MAP") == "auto" else {}),
                use_dara=use_dara,
                max_images=int(training_args.train_scenario)+1,
                vis_token_amplifier=training_args.vis_token_amplifier,
            )
            model.language_model.model.add_mimic_module(
                enable_attn_shift=True,
                attn_shift_multihead=True,
                learnable_mu=True,
                keep_logZ1_fp32=True,
            )
        elif attn_implementation == "eager":
            model = InternVLChatModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                use_flash_attn=False,
                trust_remote_code=True,
                **({"device_map": split_model(None, model_args.model_name_or_path)} if os.environ.get("P40_DEVICE_MAP") == "auto" else {}),
                use_dara=use_dara,
                max_images=int(training_args.train_scenario)+1,
                vis_token_amplifier=training_args.vis_token_amplifier,
            )
        else:
            # if training_args.mode == "grpo":
            #     model = AutoModelForImageTextToText.from_pretrained(
            #         model_args.model_name_or_path,
            #         torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
            #         low_cpu_mem_usage=True,
            #         attn_implementation=attn_implementation,
            #         trust_remote_code=True)
            # else:
            model = InternVLChatModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                # use_flash_attn=True,
            trust_remote_code=True,
            **({"device_map": split_model(None, model_args.model_name_or_path)} if os.environ.get("P40_DEVICE_MAP") == "auto" else {}))
        # model = AutoModel.from_pretrained(
        #     model_args.model_name_or_path,
        #     torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
        #     low_cpu_mem_usage=True,
        #     use_flash_attn=True,
        #     trust_remote_code=True).eval().cuda()
        
        if training_args.mode == "dara":
            model.add_dara_module(max_images=int(training_args.train_scenario)+1)   

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False, add_eos_token=False)
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    
    
    elif 'intern' in model_args.model_name_or_path.lower():
        from models.internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
        from models.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = 0.01
        config.llm_config.attn_implementation = attn_implementation  # for InternLM
        config.template = "internvl2_5"
        config.select_layer = -1
        config.dynamic_image_size = True
        config.use_thumbnail = False
        config.ps_version = 'v2'
        config.min_dynamic_patch = 1
        config.max_dynamic_patch = 6
        
        if attn_implementation == "eager":
            model = InternVLChatModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                use_dara=use_dara,
                max_images=int(training_args.train_scenario)+1)
        else:
            model = InternVLChatModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False, add_eos_token=False)
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    
    elif 'vl' in model_args.model_name_or_path:
        from models.llava.llava_multi import LlavaMultiForConditionalGeneration
        if attn_implementation == "eager":
            model = LlavaMultiForConditionalGeneration.from_pretrained( # LlavaForConditionalGeneration
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
            use_flash_attention_2=False,
            # token=ACCESS_TOKEN
        )   
        else:
            model = LlavaMultiForConditionalGeneration.from_pretrained( # LlavaForConditionalGeneration
                model_args.model_name_or_path,
                torch_dtype=compute_dtype,
                use_flash_attention_2=True,
                # token=ACCESS_TOKEN
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        data_args.image_processor = processor.image_processor
      
    elif 'qwen2.5' in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        from models.qwen.modeling_qwen2_5_vl import My_Qwen2_5_VLForConditionalGeneration

        model =  My_Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
            use_dara=use_dara,
            use_only=use_only,
            max_images=int(training_args.train_scenario)+1,
            **bnb_model_from_pretrained_args
        )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                padding_side="right")
        
        
    elif 'qwen3' in model_args.model_name_or_path.lower():
        from models.qwenvl3.modeling_qwen3_vl import My_Qwen3VLForConditionalGeneration
        from models.qwenvl3.modeling_qwen3_vl_mimic import My_Qwen3VLForConditionalGeneration_mimic
        from models.qwenvl3.modeling_qwen3_vl_dara import My_Qwen3VLForConditionalGeneration_dara
        
        if use_vis_token_amplifier or use_pai:
            model =  My_Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                # use_only=use_only,
                use_pai=use_pai,
                max_images=int(training_args.train_scenario)+1,
                vis_token_amplifier=training_args.vis_token_amplifier,
                **bnb_model_from_pretrained_args
            )
        elif use_icot:
            model =  My_Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                use_icot=use_icot,
                max_images=int(training_args.train_scenario)+1,
                **bnb_model_from_pretrained_args
            )
        elif training_args.mode == "mimic":
            model =  My_Qwen3VLForConditionalGeneration_mimic.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                use_dara=use_dara,
                # use_only=use_only,
                max_images=int(training_args.train_scenario)+1,
                **bnb_model_from_pretrained_args
            )
            model.model.language_model.add_mimic_module(
                enable_attn_shift=True,
                attn_shift_multihead=True,
                learnable_mu=True,
                keep_logZ1_fp32=True,
            )
        elif training_args.mode == "dara":
            model =  My_Qwen3VLForConditionalGeneration_dara.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                # attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                use_dara=use_dara,
                # use_only=use_only,
                max_images=int(training_args.train_scenario)+1,
                **bnb_model_from_pretrained_args
            )
        else:
            model =  My_Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
                use_dara=use_dara,
                # use_only=use_only,
                max_images=int(training_args.train_scenario)+1,
                **bnb_model_from_pretrained_args
            )
        
        if training_args.mode=="dara":
            model.model.language_model.add_dara_module(
                max_images=int(training_args.train_scenario)+1,           
                init_value=1.0,
            )
        
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                padding_side="right")
        
    elif 'otter' in model_args.model_name_or_path.lower():
        from models.otter.models.otter.modeling_otter import OtterForConditionalGeneration
        from models.otter.models.otter.org_modeling_otter import Org_OtterForConditionalGeneration

        # # if use_dara:
        model = OtterForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)),
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            use_dara=use_dara,max_images=int(training_args.train_scenario)+1,
            **bnb_model_from_pretrained_args
        )
        tokenizer = model.text_tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if training_args.is_eval:
            tokenizer.padding_side = "left"
            
        data_args.image_processor = CLIPImageProcessor.from_pretrained(model_args.vision_tower, cache_dir=training_args.cache_dir)
        data_args.is_multimodal = True

        model_args.mm_use_im_start_end = False
        model_args.mm_use_im_patch_token = False


    if 'intern' in model_args.model_name_or_path.lower() or 'vl' in model_args.model_name_or_path.lower() or 'onevision' in model_args.model_name_or_path.lower() or 'otter' in model_args.model_name_or_path.lower():
        model.config.use_cache = False
        model.requires_grad_(True)
    else:
        model.config.use_cache = False
        model.model.requires_grad_(True)
    
    if use_dara:
        for n,p in model.named_parameters():
            if "amplifier" not in n:
                p.requires_grad = False
    elif training_args.mode == "mimic":
        for n,p in model.named_parameters():
            if "log_Z1_lin" not in n and "attn_shift" not in n:
                p.requires_grad = False
    else:
        for n,p in model.named_parameters():
            p.requires_grad = False
    # FIXME
    if training_args.bits >= 16:
        if len(set(getattr(model, "hf_device_map", {}).values())) > 1:
            pass  # [p40] already sharded across GPUs by device_map="auto"
        else:
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
            
    if 'qwen' not in model_args.model_name_or_path.lower() or 'onevision' in model_args.model_name_or_path.lower():
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        # target_modules = ['k_proj', 'v_proj']
        
        if 'qwen' in model_args.model_name_or_path.lower() or 'onevision' in model_args.model_name_or_path.lower() or 'otter' in model_args.model_name_or_path.lower():
            lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            )
        elif 'internvl2_5' in model_args.model_name_or_path.lower():
            if model.llm_arch_name in ['My_InternLM2ForCausalLM', 'InternLM2ForCausalLM']:
                target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
            elif model.llm_arch_name == 'Phi3ForCausalLM':
                target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
            elif model.llm_arch_name in ['My_Qwen2ForCausalLM', 'Qwen2ForCausalLM', 'LlamaForCausalLM']:
                target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                                'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']

            lora_config = LoraConfig(
            r=training_args.lora_r,
            target_modules=target_modules,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            task_type='CAUSAL_LM'
            )
        elif 'internvl3_5' in model_args.model_name_or_path.lower():
            target_modules = [
                'self_attn.q_proj',
                'self_attn.k_proj',
                'self_attn.v_proj',
                'self_attn.o_proj',
                'mlp.gate_proj',
                'mlp.up_proj',
                'mlp.down_proj',
            ]

            lora_config = LoraConfig(
                r=training_args.lora_r,
                target_modules=target_modules,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                task_type='CAUSAL_LM'
            )
        elif 'internvl3' in model_args.model_name_or_path.lower():

            if model.llm_arch_name in ['My_InternLM2ForCausalLM', 'InternLM2ForCausalLM']:
                target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
            elif model.llm_arch_name == 'Phi3ForCausalLM':
                target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
            elif model.llm_arch_name in ['My_Qwen2ForCausalLM', 'Qwen2ForCausalLM', 'LlamaForCausalLM']:
                target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                                'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
                
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
        
        if training_args.mode in ['cluster']:
            from models.duallora.dualloramodel import DualLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALLORA'] = DualLoraModel
            lora_config.peft_type = 'DUALLORA'
        # model.enable_input_require_grads()
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    
    if training_args.lora_enable and 'internvl3_5' in model_args.model_name_or_path.lower():
        target_model = model.base_model.model 
        
        original_base_forward = target_model.forward

        def fixed_forward(*args, **kwargs):
            if 'inputs_embeds' in kwargs:
                kwargs.pop('inputs_embeds')
            if 'position_ids' in kwargs:                # <-- add
                kwargs.pop('position_ids')  
            return original_base_forward(*args, **kwargs)

        target_model.forward = fixed_forward

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
    
    elif 'otter' in model_args.model_name_or_path.lower():
        if 'otter' in conversation_lib_llava.conv_templates:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["otter"]
        else:
            raise ValueError("Could not find 'otter' conversation template.")
            
    elif 'bunny' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_bunny.conv_templates:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates[model_args.version]
        else:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates["default"]

    if 'llava' in model_args.model_name_or_path.lower() and "qwen" not in model_args.model_name_or_path.lower() and "onevision" not in model_args.model_name_or_path.lower():
        # load vision tower
        # if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            # fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        # vision_tower.requires_grad_(True)

        data_args.image_processor = vision_tower.image_processor
        
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = "pad" #data_args.image_aspect_ratio
        
        if 'qwen' in model_args.model_name_or_path.lower() and "llava" not in model_args.model_name_or_path.lower():
            model.config.tokenizer_padding_side = processor.tokenizer.padding_side
            model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
        else:    
            model.config.tokenizer_padding_side = tokenizer.padding_side
            model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
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
            # model.activate_all()
        
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    if 'llava' in model_args.model_name_or_path.lower() and 'onevision' not in model_args.model_name_or_path.lower():
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
            print(n)
            total_count += p.numel()
    print("train_param#", total_count)
    if 'onevision' in model_args.model_name_or_path.lower():
        return model, processor.tokenizer, processor, data_args
    elif 'llava' in model_args.model_name_or_path.lower() or 'intern' in model_args.model_name_or_path.lower() or 'otter' in model_args.model_name_or_path.lower():
        return model, tokenizer, data_args
    elif 'vl' in model_args.model_name_or_path or 'idefics' in model_args.model_name_or_path.lower():
        return model, tokenizer, processor, data_args
    else:
        return model, processor, data_args

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
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix, {}, strict, [], [], [])

    for name, child in module._modules.items():
        if child is not None:
            load_deepspeed(state_dict, child, prefix + name + ".")


def get_task_vectors(model, tokenizer, datalist, training_args, data_args, global_state_dict, make_supervised_data_module):
    random.seed(training_args.seed)
    client_task_vectors = []

    data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                            tokenizer=tokenizer,
                                            data_args=copy.deepcopy(data_args))

    extra_state_dict_dict = {}
    extra_state_dict_dict['client_id']=0
    extra_state_dict_dict['curr_round']=0
    extra_state_dict_dict['fisher_freq'] = 1

    training_args.per_gpu_train_batch_size=4
    training_args.gradient_accumulation_steps=1

    trainer = fedavg_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, return_task_vectors=True)

    prev_model_param = trainer.model.state_dict()
    
    output = trainer.train()
    
    task_vectors = trainer.task_vector
    
    
    trainer.deepspeed.empty_partition_cache()
    del trainer
    
    with torch.no_grad():
        model.load_state_dict(prev_model_param, strict=False)

    return task_vectors