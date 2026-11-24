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


from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, Qwen_GenerationDataset, DataCollatorForGenerationDataset, Qwen_DataCollatorForGenerationDataset, LazySupervisedDataset, OneVision_DataCollatorForGenerationDataset, OneVision_GenerationDataset, Llama_DataCollatorForGenerationDataset,  Llama_GenerationDataset, Otter_GenerationDataset, Otter_DataCollatorForGenerationDataset, OneVision1_5_GenerationDataset, OneVision1_5_DataCollatorForGenerationDataset, QwenVL3_GenerationDataset, QwenVL3_GenerationDataset_CoT # Intern_GenerationDataset, Intern_DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num#, can_infer
from tqdm import tqdm

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava
from models.llava.builder import load_pretrained_model

import warnings
import time
import datetime
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import StoppingCriteria, StoppingCriteriaList

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image
from utils.chair import load_generated_captions, CHAIR

from baukit import TraceDict, get_module
from contextlib import nullcontext
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

ALPHABET = ['A','B','C','D','E','F']
CHOICES = {"classification": ["A)Flora", "B)Walter", "C)Trum", "D)Clare"], "nlvr2":["True", "False"]}

answerPrompt="Use the image and scene graph as context to improve the detail and clarity of the original answer: "
sgPrompt='''

For the provided image and question-answer pair, generate a scene graph in JSON format to improve the quality and/or detail of the answer. The scene graph can include the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Scene Graph:
'''

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

def consecutive_ranges(t: torch.Tensor, target_val: int):
    mask = (t == target_val)

    padded = torch.cat([torch.tensor([False], device=t.device),
                        mask,
                        torch.tensor([False], device=t.device)])

    diff = padded[1:] ^ padded[:-1]  # True where state changes

    change_idx = diff.nonzero(as_tuple=False).flatten()

    starts = change_idx[0::2]
    ends   = change_idx[1::2] - 1

    return list(zip(starts.tolist(), ends.tolist()))

def get_fv_config(model, model_name_or_path):
    config = {}
    model_name_lower = model_name_or_path.lower()
    print("model_name", model_name_lower)
    
    # 1. InternVL
    if "internvl3_5" in model_name_lower:
        llm = model.language_model.model

        attn_hook_names = []
        for name, _ in llm.named_modules():
            if name.endswith("self_attn.o_proj"):
                full_name = f"language_model.model.{name}"
                attn_hook_names.append(full_name)

        config['n_heads'] = llm.config.num_attention_heads
        config['n_layers'] = len(attn_hook_names)
        config['resid_dim'] = llm.config.hidden_size
        config['attn_hook_names'] = attn_hook_names
        config['split_idx'] = 3


    elif "intern" in model_name_lower:
        llm = model.language_model
        n_layers = llm.config.num_hidden_layers
        config['n_heads'] = llm.config.num_attention_heads
        config['resid_dim'] = llm.config.hidden_size
        config['attn_hook_names'] = [f'language_model.model.layers.{layer}.attention.wo' for layer in range(n_layers)]
        config['split_idx'] = 3

    # 2. OneVision
    elif "onevision-1.5" in model_name_lower:
        llm = model.language_model

        attn_hook_names = [
        name for name, _ in model.named_modules()
        if name.endswith("self_attn.o_proj")
    ]

        config['n_heads'] = llm.config.num_attention_heads
        config['n_layers'] = len(attn_hook_names)
        config['resid_dim'] = llm.config.hidden_size
        config['attn_hook_names'] = attn_hook_names
        config['split_idx'] = 3

    elif "onevision" in model_name_lower:
        llm = model.language_model.model
        n_layers = llm.config.num_hidden_layers
        config['n_heads'] = llm.config.num_attention_heads
        config['resid_dim'] = llm.config.hidden_size
        config['attn_hook_names'] = [f'language_model.model.layers.{layer}.self_attn.o_proj' for layer in range(n_layers)]
        config['split_idx'] = 3

    # 3. Qwen
    elif "qwen3" in model_name_lower:
        llm = model

        attn_hook_names = [
            name for name, _ in model.named_modules()
            if name.endswith("self_attn.o_proj")
        ]

        config['n_heads'] = llm.config.text_config.num_attention_heads
        config['n_layers'] = len(attn_hook_names)
        config['resid_dim'] = llm.config.text_config.hidden_size
        config['attn_hook_names'] = attn_hook_names
        config['split_idx'] = 3

    elif "qwen" in model_name_lower:
        try:
            base_model = model.model
        except AttributeError:
            base_model = model
            
        n_layers = base_model.config.num_hidden_layers
        config['n_heads'] = base_model.config.num_attention_heads
        config['resid_dim'] = base_model.config.hidden_size
        config['attn_hook_names'] = [f'model.layers.{layer}.self_attn.o_proj' for layer in range(n_layers)]
        config['split_idx'] = 2

    # 4. Idefics
    elif "idefics" in model_name_lower:
        attn_hook_names = [
            name for name, _ in model.named_modules()
            if name.endswith("self_attn.o_proj")
        ]
        
        text_config = model.config.text_config
        n_layers = text_config.num_hidden_layers
        config['n_heads'] = text_config.num_attention_heads
        config['n_layers'] = len(attn_hook_names)
        config['resid_dim'] = text_config.hidden_size
        config['attn_hook_names'] = [f'model.text_model.layers.{layer}.self_attn.o_proj' for layer in range(n_layers)]
        config['split_idx'] = 3

    # 5. Otter
    elif "otter" in model_name_lower:
        lang_model = getattr(model, 'lang_encoder', model)
        n_layers = lang_model.config.num_hidden_layers
        config['n_heads'] = lang_model.config.num_attention_heads
        config['resid_dim'] = lang_model.config.hidden_size
        config['attn_hook_names'] = [f'lang_encoder.model.layers.{layer}.decoder_layer.self_attn.o_proj' for layer in range(n_layers)]
        config['split_idx'] = 3
        
    return config


def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, fv_config, model_args):
    """
    Adds (or replaces, depending on how you interpret it) the last-token head input with avg activations.

    avg_activations can be:
      - (n_layers, n_heads, head_dim)            OR
      - (n_layers, n_heads, 1, head_dim)         (backward compatible with your earlier code)
    """
    edit_layers = [x[0] for x in layer_head_token_pairs]

    def _get_act(avg_acts: torch.Tensor, layer: int, head: int) -> torch.Tensor:
        # supports (L,H,D) or (L,H,1,D)
        if avg_acts.dim() == 4:
            return avg_acts[layer, head, 0]
        return avg_acts[layer, head]

    def rep_act(output, layer_name, inputs):
        if isinstance(inputs, tuple):
            inp = inputs[0]
        else:
            inp = inputs

        # if seq len == 1, there's no "last token" to edit meaningfully in your setting
        if inp.shape[1] == 1:
            return output

        parts = layer_name.split(".")
        try:
            current_layer = int(parts[fv_config["split_idx"]])
        except (IndexError, ValueError):
            current_layer = None
            for p in parts:
                if p.isdigit():
                    current_layer = int(p)
                    break
            if current_layer is None:
                return output

        if current_layer not in edit_layers:
            return output

        original_shape = inp.shape  # (B, T, resid_dim)

        # (B, T, n_heads, head_dim)
        new_shape = inp.size()[:-1] + (fv_config["n_heads"], fv_config["resid_dim"] // fv_config["n_heads"])
        inp_reshaped = inp.view(*new_shape)

        # apply edits
        for (layer, head_n, _) in layer_head_token_pairs:
            if layer == current_layer:
                act_val = _get_act(avg_activations, layer, head_n).to(inp.device).to(inp.dtype)
                # NOTE: this is "add" (yours). If you want "replace", set "=" instead of "+="
                inp_reshaped[:, -1, head_n, :] = inp_reshaped[:, -1, head_n, :] + act_val

        inp = inp_reshaped.view(*original_shape)

        proj_module = get_module(model, layer_name)
        out_proj = proj_module.weight

        new_output = torch.matmul(inp, out_proj.T)
        return new_output

    return rep_act


# =========================
# 2) HELPERS FOR INDIRECT EFFECT IN VLM SETTING
# =========================
def _first_token_id(text: str, tokenizer) -> Optional[int]:
    """
    Returns first non-BOS token id of `text`.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        return None
    return int(ids[0])


@torch.no_grad()
def _forward_last_logits_for_batch(
    model,
    model_args,
    batch: Dict[str, Any],
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns logits of the LAST position: shape (B, vocab)
    This tries to mirror the kwargs used in your generate() calls.

    IMPORTANT:
    - This assumes batch_size==1 works (your eval uses batch_size=1).
    - If your batch format differs per dataset/collator, you may need to add a small branch here.
    """
    # Common
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)

    # Move
    if isinstance(input_ids, (list, tuple)):
        # some collators pack as list; your code often uses inputs[0]
        input_ids_t = input_ids[0].to(device)
    else:
        input_ids_t = input_ids.to(device)

    if attention_mask is not None:
        if isinstance(attention_mask, (list, tuple)):
            attention_mask_t = attention_mask[0].to(device)
        else:
            attention_mask_t = attention_mask.to(device)
    else:
        attention_mask_t = None

    model_name = model_args.model_name_or_path

    # -------- Qwen3-VL branch (you use pixel_values + image_grid_thw) --------
    if "Qwen3" in model_name:
        pixel_values = batch["pixel_values"].to(device)
        image_grid_thw = batch["image_grid_thw"]
        if isinstance(image_grid_thw, (list, tuple)):
            image_grid_thw = image_grid_thw[0].to(device)
        else:
            image_grid_thw = image_grid_thw.to(device)

        out = model(
            input_ids_t,
            attention_mask=attention_mask_t,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        logits = out.logits  # (B, T, V)
        return logits[:, -1, :]

    # -------- OneVision (llava-style) branch --------
    if "onevision" in model_name.lower():
        pixel_values = batch["pixel_values"].to(device)
        image_grid_thw = batch.get("image_grid_thw", None)
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, (list, tuple)):
                image_grid_thw = image_grid_thw[0].to(device)
            else:
                image_grid_thw = image_grid_thw.to(device)

        # some OneVision variants use images/pixel_values depending on your collator;
        # your generate() used pixel_values + image_grid_thw for onevision-1.5
        kwargs = dict(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            pixel_values=pixel_values,
            use_cache=False,
        )
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw

        out = model(**kwargs)
        return out.logits[:, -1, :]

    # -------- Llava/VL branch (if your collator uses 'pixel_values' or 'images') --------
    if ("llava" in model_name.lower()) or ("vl" in model_name.lower()):
        # in your eval code you sometimes pass images=imgs, sometimes pixel_values=imgs
        if "pixel_values" in batch:
            imgs = batch["pixel_values"]
            kwargs = dict(input_ids=input_ids_t, attention_mask=attention_mask_t, pixel_values=imgs.to(device), use_cache=False)
        else:
            imgs = batch["images"]
            kwargs = dict(input_ids=input_ids_t, attention_mask=attention_mask_t, images=imgs.to(device), use_cache=False)

        out = model(**kwargs)
        return out.logits[:, -1, :]

    # -------- Otter branch (vision_x/lang_x) --------
    if "otter" in model_name.lower():
        vision_x = batch["pixel_values"].to(device)
        lang_x = input_ids_t
        out = model(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask_t,
            use_cache=False,
        )
        return out.logits[:, -1, :]

    # -------- fallback (text-only or unknown wrapper) --------
    out = model(input_ids_t, attention_mask=attention_mask_t, use_cache=False)
    return out.logits[:, -1, :]


@torch.no_grad()
def compute_indirect_effect_vlm(
    dataloader: DataLoader,
    mean_head_activations: torch.Tensor,  # (L,H,D) from your extract_fv
    model,
    model_args,
    tokenizer,
    device: torch.device,
    fv_config: dict,
    n_trials: int = 25,
    last_token_only: bool = True,  # kept for parity with Code 1; this impl is last-token-only
) -> torch.Tensor:
    """
    Indirect Effect (last-token-only): for each (layer, head),
    run baseline probs at last token; then add mean head activation to last-token head input
    and measure delta prob of the "answer token of interest".

    Returns:
      indirect_effect: (n_trials, L, H)  (matching Code 1 last_token_only=True)
    """
    assert last_token_only, "This VLM version currently implements last_token_only=True only."

    L = fv_config["n_layers"]
    H = fv_config["n_heads"]

    indirect_effect = torch.zeros(n_trials, L, H, dtype=torch.float32)

    # make an iterator you can cycle
    it = iter(dataloader)

    for t in tqdm(range(n_trials), total=n_trials, desc="IndirectEffect(VLM)"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)

        # token of interest: use first token of gold string (batch["gold"])
        golds = batch.get("gold", None)
        if golds is None:
            # cannot compute without target; skip
            continue
        gold_text = golds[0] if isinstance(golds, (list, tuple)) else golds
        tok_id = _first_token_id(str(gold_text), tokenizer)
        if tok_id is None:
            continue

        # Baseline
        clean_logits = _forward_last_logits_for_batch(model, model_args, batch, tokenizer, device)
        clean_probs = torch.softmax(clean_logits, dim=-1)  # (B,V)
        clean_p = clean_probs[0, tok_id].item()

        # Sweep heads
        for layer in range(L):
            hook_name = fv_config["attn_hook_names"][layer]

            for head in range(H):
                intervention_locations = [(layer, head, -1)]
                intervention_fn = last_replace_activation_w_avg(
                    layer_head_token_pairs=intervention_locations,
                    avg_activations=mean_head_activations,
                    model=model,
                    fv_config=fv_config,
                    model_args=model_args,
                )

                with TraceDict(model, layers=[hook_name], edit_output=intervention_fn, retain_input=False, retain_output=False):
                    inter_logits = _forward_last_logits_for_batch(model, model_args, batch, tokenizer, device)

                inter_probs = torch.softmax(inter_logits, dim=-1)
                inter_p = inter_probs[0, tok_id].item()

                indirect_effect[t, layer, head] = float(inter_p - clean_p)

    return indirect_effect


def extract_fv(
    dataset,
    dataname,
    round,
    model,
    tokenizer,
    device,
    model_args,
    training_args,
    logger,
    client_id=None,
    batch_size=1,
    shot_num=1,
    data=None,
    processor=None,
    fv_data=None,
    # ---- NEW knobs ----
    compute_ie=True,
    topk_heads=20,
    save_ie_per_sample=False,
):
    """
    Keeps your original 3 branches:
      - idefics
      - intern
      - generic dataloader (llava/qwen/onevision/vl/otter/etc.)

    Adds:
      - mean head activations (last token, input-to-o_proj) saved as (L,H,D)
      - mean indirect effect (IE) saved as (L,H) when logits are available
      - topK heads by |mean IE|
      - sum of those heads' mean activations saved as (D) (+ mean as (D))
    """
    import os
    import json
    import torch
    import numpy as np
    from tqdm import tqdm
    from contextlib import nullcontext
    from baukit import TraceDict, get_module

    # -----------------------------
    # paths
    # -----------------------------
    out_dir = (
        f"./fv/{training_args.ordering}/{training_args.eval_shot}/"
        f"{training_args.dataset_name}/scenario{training_args.scenario}/"
        f"{model_args.model_name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # config for hooks
    # -----------------------------
    fv_cfg = get_fv_config(model, model_args.model_name_or_path)
    hook_names = fv_cfg.get("attn_hook_names", [])
    if not hook_names:
        raise RuntimeError(f"No attn_hook_names found for {model_args.model_name_or_path}. Check get_fv_config().")

    n_layers = fv_cfg["n_layers"]
    n_heads  = fv_cfg["n_heads"]
    resid_dim = fv_cfg["resid_dim"]
    head_dim = resid_dim // n_heads

    # -----------------------------
    # intervention fn (same semantics as your last_replace_activation_w_avg)
    # -----------------------------
    def _get_current_layer(layer_name: str, split_idx: int):
        parts = layer_name.split(".")
        try:
            return int(parts[split_idx])
        except Exception:
            for p in parts:
                if p.isdigit():
                    return int(p)
        return None

    def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, fv_config, model_args):
        edit_layers = set([x[0] for x in layer_head_token_pairs])

        def rep_act(output, layer_name, inputs):
            inp = inputs[0] if isinstance(inputs, tuple) else inputs
            if inp is None:
                return output
            if inp.shape[1] == 1:
                return output

            current_layer = _get_current_layer(layer_name, fv_config["split_idx"])
            if current_layer is None or current_layer not in edit_layers:
                return output

            B, T, D = inp.shape
            H = fv_config["n_heads"]
            hd = fv_config["resid_dim"] // H

            inp_h = inp.view(B, T, H, hd)

            for (layer, head_n, _tok) in layer_head_token_pairs:
                if layer == current_layer:
                    act_val = avg_activations[layer, head_n].to(inp.device, dtype=inp.dtype)  # (hd,)
                    inp_h[:, -1, head_n, :] = inp_h[:, -1, head_n, :] + act_val

            inp_new = inp_h.view(B, T, D)

            proj_module = get_module(model, layer_name)
            W = proj_module.weight
            new_output = torch.matmul(inp_new, W.T)
            return new_output

        return rep_act

    # -----------------------------
    # mean head activation accumulator
    # -----------------------------
    @torch.no_grad()
    def _accumulate_last_token_head_acts(td, cfg: dict, act_sum: torch.Tensor, act_count: int):
        # (L,B,T,D)
        layer_inp = torch.stack([td[name].input for name in cfg["attn_hook_names"]], dim=0)
        H = cfg["n_heads"]
        hd = cfg["resid_dim"] // H
        layer_inp = layer_inp.view(cfg["n_layers"], *layer_inp.shape[1:-1], H, hd)  # (L,B,T,H,hd)
        last_tok = layer_inp[:, :, -1, :, :]  # (L,B,H,hd)
        act_sum += last_tok.sum(dim=1).detach().to(act_sum.device, dtype=act_sum.dtype)  # sum over B
        act_count += last_tok.shape[1]
        return act_sum, act_count

    # -----------------------------
    # helper: first token id of gold string
    # -----------------------------
    def _first_token_id(text: str):
        ids = tokenizer(str(text), add_special_tokens=False).input_ids
        return None if len(ids) == 0 else int(ids[0])

    # -----------------------------
    # buffers
    # -----------------------------
    act_sum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float32, device="cpu")
    act_count = 0

    ie_sum = torch.zeros(n_layers, n_heads, dtype=torch.float64, device="cpu")
    ie_count = 0
    ie_per_sample = [] if save_ie_per_sample else None

    model = model.eval()

    # -----------------------------
    # a unified "TraceDict context" builder
    # -----------------------------
    def _make_traced_context(retain_input: bool, edit_output_fn=None, layers_override=None):
        layers = layers_override if layers_override is not None else hook_names
        if edit_output_fn is None:
            return TraceDict(model, layers=layers, retain_input=retain_input, retain_output=False)
        return TraceDict(model, layers=layers, retain_input=retain_input, retain_output=False, edit_output=edit_output_fn)

    # -----------------------------
    # IE computation given fw_kwargs and gold_token_id
    # -----------------------------
    # @torch.no_grad()
    # def _compute_ie_for_one_sample(fw_kwargs: dict, gold_token_id: int, mean_head_activations: torch.Tensor):
    #     """
    #     IE(layer,head) = p_int(gold) - p_clean(gold), where intervention adds mean head activation
    #     to that head at last token input to o_proj.
    #     Returns (L,H) float64 on CPU.
    #     """
    #     # clean
    #     out = model(**fw_kwargs).logits
    #     clean_logits = out[:, -1, :]
    #     clean_probs = torch.softmax(clean_logits, dim=-1)
    #     clean_p = float(clean_probs[0, gold_token_id].item())

    #     ie = torch.zeros(n_layers, n_heads, dtype=torch.float64)

    #     # per head intervention
    #     for layer in range(n_layers):
    #         layer_hook = hook_names[layer]
    #         for head in range(n_heads):
    #             locations = [(layer, head, -1)]
    #             edit_fn = last_replace_activation_w_avg(
    #                 layer_head_token_pairs=locations,
    #                 avg_activations=mean_head_activations,  # CPU tensor ok (copied in fn)
    #                 model=model,
    #                 fv_config=fv_cfg,
    #                 model_args=model_args,
    #             )
    #             with _make_traced_context(retain_input=False, edit_output_fn=edit_fn, layers_override=[layer_hook]):
    #                 out_i = model(**fw_kwargs).logits
    #             probs_i = torch.softmax(out_i[:, -1, :], dim=-1)
    #             p_i = float(probs_i[0, gold_token_id].item())
    #             ie[layer, head] = (p_i - clean_p)

    #     return ie
    @torch.no_grad()
    def _compute_ie_for_one_sample(
        fw_kwargs: dict,
        gold_token_id: int,
        mean_head_activations: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """
        GPU version.
        Returns: ie (L, H) on GPU, float32 (or float16/bf16 if you prefer).
        """

        # Ensure fw_kwargs tensors are on GPU
        fw_dev = {}
        for k, v in fw_kwargs.items():
            fw_dev[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v

        # Put mean activations on GPU once (keep as fp16/bf16 if you want speed)
        # mean_head_activations expected shape: (L, H, head_dim)
        mean_head_acts_dev = mean_head_activations.to(device=device, dtype=torch.float16, non_blocking=True)

        # ---- clean ----
        out = model(**fw_dev).logits                         # (B, T, V)
        clean_logits = out[:, -1, :]                         # (B, V)

        # Get clean prob of the gold token WITHOUT full softmax:
        # p = exp(logit_g - logsumexp(logits))
        clean_logit_g = clean_logits[:, gold_token_id]       # (B,)
        clean_logZ = torch.logsumexp(clean_logits, dim=-1)   # (B,)
        clean_p = torch.exp(clean_logit_g - clean_logZ)      # (B,) on GPU

        ie = torch.zeros((n_layers, n_heads), device=device, dtype=torch.float32)

        # ---- per head intervention ----
        for layer in range(n_layers):
            layer_hook = hook_names[layer]

            for head in range(n_heads):
                locations = [(layer, head, -1)]
                edit_fn = last_replace_activation_w_avg(
                    layer_head_token_pairs=locations,
                    avg_activations=mean_head_acts_dev,   # NOW GPU
                    model=model,
                    fv_config=fv_cfg,
                    model_args=model_args,
                )

                with _make_traced_context(
                    retain_input=False,
                    edit_output_fn=edit_fn,
                    layers_override=[layer_hook],
                ):
                    out_i = model(**fw_dev).logits

                inter_logits = out_i[:, -1, :]                       # (B, V)
                inter_logit_g = inter_logits[:, gold_token_id]       # (B,)
                inter_logZ = torch.logsumexp(inter_logits, dim=-1)   # (B,)
                inter_p = torch.exp(inter_logit_g - inter_logZ)      # (B,) on GPU

                ie[layer, head] = (inter_p - clean_p).float()[0]     # assumes B==1

        return ie

    @torch.no_grad()
    def _compute_ie_for_one_sample_intern(
        question = None,
        pixel_values = None,
        generation_config = None,
        num_patches_list = None, 
        gold_token_id = None,
        mean_head_activations = None,
        *,
        device = None,
    ) -> torch.Tensor:
        """
        GPU version.
        Returns: ie (L, H) on GPU, float32 (or float16/bf16 if you prefer).
        """

        # # Ensure fw_kwargs tensors are on GPU
        # fw_dev = {}
        # for k, v in fw_kwargs.items():
        #     fw_dev[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v

        # Put mean activations on GPU once (keep as fp16/bf16 if you want speed)
        # mean_head_activations expected shape: (L, H, head_dim)
        mean_head_acts_dev = mean_head_activations.to(device=device, dtype=torch.float16, non_blocking=True)

        # ---- clean ----
        # out = model(**fw_dev).logits                         # (B, T, V)
        out = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            output_hidden_state=True,
            history=None,
            return_history=True,
        )
        clean_logits = out[:, -1, :]                         # (B, V)

        # Get clean prob of the gold token WITHOUT full softmax:
        # p = exp(logit_g - logsumexp(logits))
        clean_logit_g = clean_logits[:, gold_token_id]       # (B,)
        clean_logZ = torch.logsumexp(clean_logits, dim=-1)   # (B,)
        clean_p = torch.exp(clean_logit_g - clean_logZ)      # (B,) on GPU

        ie = torch.zeros((n_layers, n_heads), device=device, dtype=torch.float32)

        # ---- per head intervention ----
        for layer in range(n_layers):
            layer_hook = hook_names[layer]

            for head in range(n_heads):
                locations = [(layer, head, -1)]
                edit_fn = last_replace_activation_w_avg(
                    layer_head_token_pairs=locations,
                    avg_activations=mean_head_acts_dev,   # NOW GPU
                    model=model,
                    fv_config=fv_cfg,
                    model_args=model_args,
                )

                with _make_traced_context(
                    retain_input=False,
                    edit_output_fn=edit_fn,
                    layers_override=[layer_hook],
                ):
                    out_i =  model.chat(
                        tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        output_hidden_state=True,
                        history=None,
                        return_history=True,
                    )

                inter_logits = out_i[:, -1, :]                       # (B, V)
                inter_logit_g = inter_logits[:, gold_token_id]       # (B,)
                inter_logZ = torch.logsumexp(inter_logits, dim=-1)   # (B,)
                inter_p = torch.exp(inter_logit_g - inter_logZ)      # (B,) on GPU

                ie[layer, head] = (inter_p - clean_p).float()[0]     # assumes B==1

        return ie

    # ============================================================
    # BRANCH 1: IDEFICS
    # ============================================================
    if "idefics" in model_args.model_name_or_path.lower():
        assert processor is not None, "IDEFICS requires `processor`."
        mean_head_acts_cpu = None

        cache_for_ie = []

        for d in tqdm(data, desc=f"[idefics] {dataname}"):
            qs = d["conversations"][0]["value"].replace("\n", " ").strip()
            gold = d["conversations"][1]["value"]

            # parse images (your original logic condensed)
            image_file = d.get("image", None)
            if image_file is None:
                continue
            if isinstance(image_file, str):
                image_file = [image_file]

            # build the prompts list similar to your original
            prompts = []
            if "<image> <image>" in qs:
                double_image = True
                chunks = [c.strip() for c in qs.split("<image> <image") if c.strip()]
                if not qs.startswith("<image> <image"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]
            else:
                double_image = False
                chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
                if not qs.startswith("<image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]

            pairs = []
            for chunk in chunks:
                m = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                if not m:
                    m = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                if m:
                    q, a = m.groups()
                else:
                    q, a = chunk, ""
                pairs.append((q.strip(), a.strip()))

            if double_image:
                for pair_i, (q, a) in enumerate(pairs):
                    if pair_i == len(pairs) - 1:
                        prompts.append({"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": f"User: {q} \nAssistant: "}]})
                    else:
                        prompts.append({"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": f"User: {q} \nAssistant: {a}"}]})
            else:
                for i_img, (q, a) in enumerate(pairs[:len(image_file)]):
                    if i_img == len(image_file) - 1:
                        prompts.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"User: {q} \nAssistant: "}]})
                    else:
                        prompts.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"User: {q} \nAssistant: {a}"}]})

            new_image_file = [Image.open(p).convert("RGB") for p in image_file]
            prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)

            inputs = processor(text=prompt, images=new_image_file, return_tensors="pt")
            inputs = {k: v.to(device=device) for k, v in inputs.items()}

            gold_tok = _first_token_id(gold)

            # Trace for mean activations (+ optional fv intervention)
            if fv_data is not None:
                edit_fn = last_replace_activation_w_avg(
                    fv_data["locations"], fv_data["activations"], model, fv_data["config"], model_args
                )
                trace_td = _make_traced_context(retain_input=True, edit_output_fn=edit_fn, layers_override=fv_data["config"]["attn_hook_names"])
                cfg_for_td = fv_data["config"]
            else:
                trace_td = _make_traced_context(retain_input=True, edit_output_fn=None, layers_override=hook_names)
                cfg_for_td = fv_cfg

            with torch.inference_mode():
                with trace_td as td:
                    _ = model(**inputs)  # forward
                act_sum, act_count = _accumulate_last_token_head_acts(td, cfg_for_td, act_sum, act_count)

            if compute_ie and gold_tok is not None:
                cache_for_ie.append((inputs, gold_tok))

        # finalize mean activations
        mean_head_acts = (act_sum / float(act_count)).to(torch.float32)
        torch.save(mean_head_acts, os.path.join(out_dir, "mean_head_activations.pt"))
        logger.info(f"[ACT] Saved mean_head_activations.pt | shape={tuple(mean_head_acts.shape)}")

        # compute IE if requested
        if compute_ie and len(cache_for_ie) > 0:
            for inputs, gold_tok in tqdm(cache_for_ie, desc="[idefics] IE"):
                fw_kwargs = dict(**inputs)
                ie = _compute_ie_for_one_sample(fw_kwargs, gold_tok, mean_head_acts, device="cuda").cpu()
                ie_sum += ie
                ie_count += 1
                if save_ie_per_sample:
                    ie_per_sample.append(ie)

    # ============================================================
    # BRANCH 2: INTERN (InternVL / Intern)
    # ============================================================
    elif "intern" in model_args.model_name_or_path.lower():
        cache_for_ie = []

        for d in tqdm(data, desc=f"[intern] {dataname}"):
            if "image" not in d:
                continue

            # prep images as your original
            pixel_values_list = []
            num_patches_list = []
            images = d["image"]
            if isinstance(images, str):
                images = [images]
            for image in images:
                pv = load_image(image, max_num=12).to(torch.bfloat16).to(device=device)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.size(0))
            pixel_values = torch.cat(pixel_values_list, dim=0)

            question = d["conversations"][0]["value"]
            gold = d["conversations"][1]["value"]
            gold_tok = _first_token_id(gold)

            # Trace for mean activations (+ optional fv)
            if fv_data is not None:
                edit_fn = last_replace_activation_w_avg(
                    fv_data["locations"], fv_data["activations"], model, fv_data["config"], model_args
                )
                trace_td = _make_traced_context(retain_input=True, edit_output_fn=edit_fn, layers_override=fv_data["config"]["attn_hook_names"])
                cfg_for_td = fv_data["config"]
            else:
                trace_td = _make_traced_context(retain_input=True, edit_output_fn=None, layers_override=hook_names)
                cfg_for_td = fv_cfg

            # Intern chat (generation). This does not naturally give logits.
            generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode != "dara")
            with torch.inference_mode():
                with trace_td as td:
                    _resp, _hist = model.chat(
                        tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True,
                    )
                act_sum, act_count = _accumulate_last_token_head_acts(td, cfg_for_td, act_sum, act_count)

            # Best-effort IE: try to obtain logits via language_model forward if exists
            if compute_ie and gold_tok is not None:
                cache_for_ie.append((pixel_values, num_patches_list, question, gold_tok))

        # finalize mean activations
        mean_head_acts = (act_sum / float(act_count)).to(torch.float32).cpu()
        torch.save(mean_head_acts, os.path.join(out_dir, "mean_head_activations.pt"))
        logger.info(f"[ACT] Saved mean_head_activations.pt | shape={tuple(mean_head_acts.shape)}")

        # Try IE only if we can do a forward that returns logits.
        # Many Intern wrappers don't expose a simple multimodal forward; if absent, we skip IE cleanly.
        can_do_logits = hasattr(model, "forward") and ("language_model" in dir(model) or "language_model" in model.__dict__)
        if compute_ie and can_do_logits and len(cache_for_ie) > 0:
            logger.info("[IE][intern] Attempting best-effort logits forward for IE. If this fails, IE will be skipped.")
            for pixel_values, num_patches_list, question, gold_tok in tqdm(cache_for_ie, desc="[intern] IE"):
                try:
                    # If your Intern model supports a direct forward with prepared inputs, adapt here.
                    # Minimal fallback: tokenize question as text-only; this is NOT perfect multimodal IE.
                    # inputs = tokenizer([question], return_tensors="pt").to(device=device)
                    # fw_kwargs = dict(**inputs)
                    ie = _compute_ie_for_one_sample_intern(question, pixel_values,generation_config, num_patches_list, gold_tok, mean_head_acts, device="cuda").cpu()
                    ie_sum += ie
                    ie_count += 1
                    if save_ie_per_sample:
                        ie_per_sample.append(ie)
                except Exception as e:
                    logger.warning(f"[IE][intern] Skipping IE for a sample due to error: {e}")
        else:
            if compute_ie:
                logger.warning("[IE][intern] Skipping IE: no reliable logits forward path available for this Intern wrapper.")

    # ============================================================
    # BRANCH 3: GENERIC DATALOADER (llava/qwen/onevision/vl/otter/etc.)
    # ============================================================
    else:
        # dataloader setup: same as your original
        if "onevision-1.5" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
                collate_fn=OneVision1_5_DataCollatorForGenerationDataset(tokenizer),
            )
        elif "onevision-qwen" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
                collate_fn=OneVision_DataCollatorForGenerationDataset(tokenizer),
            )
        elif "Qwen" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
                collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer),
            )
        elif "llava" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
                collate_fn=DataCollatorForGenerationDataset(tokenizer),
            )
        elif "vl" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
                collate_fn=Llama_DataCollatorForGenerationDataset(tokenizer),
            )
        elif "otter" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
                collate_fn=Otter_DataCollatorForGenerationDataset(tokenizer),
            )
        else:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
                collate_fn=DataCollatorForGenerationDataset(tokenizer),
            )

        cache_for_ie = []
        for batch in tqdm(dataloader, desc=f"[generic] {dataname}"):
            imgs = None
            image_grid_thw = None
            image_sizes = None

            if "Qwen" in model_args.model_name_or_path:
                inputs, imgs, golds, img_files, prompts, image_grid_thw = (
                    batch["input_ids"], batch["pixel_values"], batch["gold"], batch["image_file"], batch["prompt"], batch["image_grid_thw"]
                )
                attention_mask = batch["attention_mask"].to(device=device)[0]
                inputs = inputs.to(device=device)[0]
                imgs = imgs.to(device=device, dtype=torch.bfloat16)
            elif "onevision-1.5" in model_args.model_name_or_path.lower():
                inputs, imgs, golds, img_files, prompts, image_grid_thw = (
                    batch["input_ids"], batch["pixel_values"], batch["gold"], batch["image_file"], batch["prompt"], batch["image_grid_thw"]
                )
                attention_mask = batch["attention_mask"].to(device=device)[0]
                inputs = inputs.to(device=device)[0]
                imgs = imgs.to(device=device, dtype=torch.bfloat16)
            elif "vl" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"], batch["pixel_values"], batch["gold"], batch["prompt"], batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)
                inputs = inputs.to(device=device)
                imgs = imgs.to(device=device, dtype=torch.bfloat16)
            elif "otter" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"], batch["pixel_values"], batch["gold"], batch["prompt"], batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)
                inputs = inputs.to(device=device)
                imgs = imgs.to(device=device, dtype=torch.bfloat16)
            else:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"], batch["images"], batch["gold"], batch["prompt"], batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)
                inputs = inputs.to(device=device)
                if imgs is not None:
                    if isinstance(imgs, list):
                        imgs = [img.to(device=device, dtype=torch.bfloat16) for img in imgs]
                    else:
                        imgs = imgs.to(device=device, dtype=torch.bfloat16)
                    image_sizes = [x.shape[-2:] for x in imgs]

            gold_text = golds[0] if isinstance(golds, (list, tuple)) else golds
            gold_tok = _first_token_id(gold_text)

            # Trace for mean acts (+ optional fv)
            if fv_data is not None:
                edit_fn = last_replace_activation_w_avg(
                    fv_data["locations"], fv_data["activations"], model, fv_data["config"], model_args
                )
                trace_td = _make_traced_context(retain_input=True, edit_output_fn=edit_fn, layers_override=fv_data["config"]["attn_hook_names"])
                cfg_for_td = fv_data["config"]
            else:
                trace_td = _make_traced_context(retain_input=True, edit_output_fn=None, layers_override=hook_names)
                cfg_for_td = fv_cfg

            # forward kwargs (for logits)
            fw_kwargs = {}
            if "Qwen" in model_args.model_name_or_path or "onevision-1.5" in model_args.model_name_or_path.lower():
                fw_kwargs = dict(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    pixel_values=imgs,
                    image_grid_thw=image_grid_thw[0].cuda(),
                )
                fw_kwargs = {k: v for k, v in fw_kwargs.items() if v is not None}
            elif "onevision" in model_args.model_name_or_path.lower() and ("onevision-1.5" not in model_args.model_name_or_path.lower()):
                fw_kwargs = dict(input_ids=inputs, attention_mask=attention_mask, images=imgs, image_sizes=image_sizes)
                fw_kwargs = {k: v for k, v in fw_kwargs.items() if v is not None}
            elif "llava" in model_args.model_name_or_path.lower():
                fw_kwargs = dict(input_ids=inputs, attention_mask=attention_mask, images=imgs)
                fw_kwargs = {k: v for k, v in fw_kwargs.items() if v is not None}
            elif "otter" in model_args.model_name_or_path.lower():
                fw_kwargs = dict(vision_x=imgs, lang_x=inputs, attention_mask=attention_mask)
            else:
                fw_kwargs = dict(input_ids=inputs, attention_mask=attention_mask, pixel_values=imgs)
                fw_kwargs = {k: v for k, v in fw_kwargs.items() if v is not None}

            with torch.inference_mode():
                with trace_td as td:
                    _ = model(**fw_kwargs)
                act_sum, act_count = _accumulate_last_token_head_acts(td, cfg_for_td, act_sum, act_count)

            if compute_ie and gold_tok is not None:
                # store a lightweight copy of fw_kwargs (tensors kept; you can detach/cpu if memory is tight)
                cache_for_ie.append((fw_kwargs, gold_tok))

        # finalize mean activations
        mean_head_acts = (act_sum / float(act_count)).to(torch.float32).cpu()
        torch.save(mean_head_acts, os.path.join(out_dir, "mean_head_activations.pt"))
        logger.info(f"[ACT] Saved mean_head_activations.pt | shape={tuple(mean_head_acts.shape)}")
        # compute IE
        if compute_ie and len(cache_for_ie) > 0:
            for fw_kwargs, gold_tok in tqdm(cache_for_ie, desc="[generic] IE"):
                # ensure kwargs on device
                fw_dev = {}
                for k, v in fw_kwargs.items():
                    fw_dev[k] = v.to(device=device) if torch.is_tensor(v) else v
                ie = _compute_ie_for_one_sample(fw_dev, gold_tok, mean_head_acts, device="cuda").cpu()
                ie_sum += ie
                ie_count += 1
                if save_ie_per_sample:
                    ie_per_sample.append(ie)

    # ============================================================
    # FINALIZE: save mean IE, topK heads, sum activations
    # ============================================================
    # mean activations already saved in each branch; reload from disk to be safe
    # mean_head_acts = torch.load(os.path.join(out_dir, "mean_head_activations.pt"), map_location="cpu")

    if compute_ie and ie_count > 0:
        mean_ie = (ie_sum / float(ie_count)).to(torch.float32)  # (L,H)
        mean_ie_path = os.path.join(out_dir, "mean_indirect_effect.pt")
        torch.save(mean_ie, mean_ie_path)
        logger.info(f"[IE] Saved mean_indirect_effect.pt | shape={tuple(mean_ie.shape)}")

        if save_ie_per_sample:
            ie_stack = torch.stack(ie_per_sample, dim=0)  # (N,L,H)
            torch.save(ie_stack, os.path.join(out_dir, "indirect_effect_per_sample.pt"))
            logger.info(f"[IE] Saved indirect_effect_per_sample.pt | shape={tuple(ie_stack.shape)}")
        # topK by |mean_ie|
        abs_ie = mean_ie.abs()
        flat = abs_ie.flatten()
        K = min(topk_heads, flat.numel())
        vals, idxs = torch.topk(flat, k=K, largest=True)

        top_heads = []
        for r in range(K):
            idx = int(idxs[r].item())
            layer = idx // n_heads
            head  = idx % n_heads
            top_heads.append({
                "rank": int(r),
                "layer": int(layer),
                "head": int(head),
                "mean_ie": float(mean_ie[layer, head].item()),
                "abs_mean_ie": float(abs_ie[layer, head].item()),
            })

        with open(os.path.join(out_dir, f"top{topk_heads}_heads_by_mean_ie.json"), "w") as f:
            json.dump(top_heads, f, indent=2)
        logger.info(f"[IE] Saved top{topk_heads}_heads_by_mean_ie.json")

        # sum those heads' mean activations
        selected_sum = torch.zeros(head_dim, dtype=torch.float32)
        for h in top_heads:
            selected_sum += mean_head_acts[h["layer"], h["head"]]
        selected_mean = selected_sum / float(len(top_heads))
        torch.save(selected_sum, os.path.join(out_dir, f"top{topk_heads}_sum_activation.pt"))
        # torch.save(selected_mean, os.path.join(out_dir, f"top{topk_heads}_mean_activation.pt"))
        logger.info(f"[ACT] Saved top{topk_heads}_sum_activation.pt | shape={tuple(selected_sum.shape)}")
        # logger.info(f"[ACT] Saved top{topk_heads}_mean_activation.pt | shape={tuple(selected_mean.shape)}")
    else:
        logger.warning("[IE] Not saved: compute_ie=False or ie_count==0 (no logits path / no gold tokens).")
    torch.cuda.empty_cache()
    

def parse_choice_list(input_string):
    # Try to find the choice list in the format "Choice list:[...]"
    match = re.search(r'choose from \[(.*?)\] Learn from the demos', input_string)
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
    
    
    match = re.search(r'What is the name of this car type, choose from \[(.*?)\]', input_string)
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
    
    match = re.search(r'Choose from the following options and output ONLY the answer: \[(.*?)\]', input_string)
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
    
    match = re.search(r'choose from \[(.*?)\] Learn from the demos', input_string)
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

    matches = re.findall(r'choose from \[(.*?)\]', input_string, re.IGNORECASE)
    if matches:
        last = matches[-1]
        choices = [c.strip() for c in last.split(',')]
        return choices
    
    # print("input", input_string)
    # match = re.search(r'\[A\).*?D\)\w+\]', input_string)
    # if match:
    #     print("matched")
    #     return [f.lower() for f in match]
    
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

    os.makedirs(f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}", exist_ok=True)
    fileHandler = logging.FileHandler(f'./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}/round_{training_args.round_to_eval}.log', mode="w")
    
    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}/federated')

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    
    if training_args.mode == "only":
        print("init only")
        evolve_only_sampling()
    
    
    print("bring model")
    processor = None
    if "Qwen" in model_args.model_name_or_path:
        model, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
        tokenizer = processor.tokenizer
    elif "Intern" in model_args.model_name_or_path:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif "onevision" in model_args.model_name_or_path.lower():
        # warnings.filterwarnings("ignore")
        # pretrained = model_args.model_name_or_path
        # model_name = "llava_qwen"
        # device = "cuda"
        # device_map = "auto"
        # llava_model_args = {
        #     "multimodal": True,
        #     "attn_implementation": training_args.attn_impl
        # }
        # tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,torch_dtype=torch.bfloat16, training_args=training_args, **llava_model_args)  # Add any other thing you want to pass in llava_model_args
        # data_args.image_processor = image_processor
        # model.eval()
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
        # LlavaOnevisionForConditionalGeneration
        # model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif "vl" in model_args.model_name_or_path:
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif 'idefics' in model_args.model_name_or_path.lower():
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    else:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    print("brought model")
            
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario, shot=training_args.eval_shot, dataset_name=training_args.dataset_name, ordering=training_args.ordering)
    batch_size = 1
    logger.info(f'Evaluatiing clients and server at round {training_args.round_to_eval}')
    start_time = time.time()
    server_eval_key = []
    
    # shot_num = training_args.shot[:training_args.shot.index("-shot")]
    shot_num =""
    print(training_args.zeroshot, training_args.eval_server)
    if not training_args.zeroshot and training_args.eval_server:
        logger.info(f'load ./client_states_{training_args.note}/server_model_round{training_args.round_to_eval-1}.pth')
        server_state_dict = torch.load(f'./client_states_{training_args.note}/server_model_round{training_args.round_to_eval-1}.pth', map_location='cpu')
        model.load_state_dict(server_state_dict, strict=False)
        print("server_state_dict", server_state_dict)
        del server_state_dict
        
    if training_args.eval_server:
        test_datalist = test_datalists[0]
        for round_i, data_info in enumerate(test_datalist):
            # if round_i + 1 > training_args.round_to_eval:
            #     continue
            print(data_info['data_name'])
            if "Qwen2.5" in model_args.model_name_or_path:
                dataset = Qwen_GenerationDataset(data_info['data'], processor,tokenizer, data_args)
            elif "Qwen3" in model_args.model_name_or_path:
                if training_args.use_cotprompt:
                    dataset = QwenVL3_GenerationDataset_CoT(data_info['data'], processor,tokenizer, data_args, prompt_ver=training_args.cotprompt_ver)
                else:
                    dataset = QwenVL3_GenerationDataset(data_info['data'], processor,tokenizer, data_args)
            elif "vl" in model_args.model_name_or_path:
                if "qwen" in model_args.model_name_or_path:
                    dataset = Llama_GenerationDataset(data_info['data'], tokenizer, data_args, processor, qwen=True)
                elif 'llama' in model_args.model_name_or_path:
                    dataset = Llama_GenerationDataset(data_info['data'], tokenizer, data_args, processor)
            elif "onevision-1.5" in model_args.model_name_or_path.lower():
                dataset = OneVision1_5_GenerationDataset(data_info['data'], tokenizer, processor, data_args)
            elif "onevision" in model_args.model_name_or_path.lower():
                dataset = OneVision_GenerationDataset(data_info['data'], tokenizer, processor, data_args)
            elif "otter" in model_args.model_name_or_path.lower():
                dataset = Otter_GenerationDataset(data_info['data'], tokenizer, data_args)
            else:
                dataset = GenerationDataset(data_info['data'], tokenizer, data_args)
            
            extract_fv(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, None, batch_size, shot_num, data_info['data'], processor=processor, fv_data=None)

        return
    
    logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    
def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values    
    
def get_datalists(args, scenario_num, shot, dataset_name, ordering):
    with open(f"./scenarios/original/{dataset_name}/scenario-2.json") as fp:
        scenario = json.load(fp)
    assert args.num_clients == len(scenario)
    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for client_data in scenario:
        client_id = client_data['client_id']
        # train_datalist = []
        test_datalist = []
        eval_cnt = 0
        # train_cnt = 0
        for data in client_data['datasets']:
            with open(f"./datalists/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            if len(datalist) > 25:
                datalist = random.sample(datalist, k=25)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "type": data['type'] if 'type' in data else 'open-ended',
                "data": datalist,
                "eval_cnt": eval_cnt})
            eval_cnt += len(datalist)
            print("total eval set len", len(datalist))
            # train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist

    return "_", test_datalists

if __name__ == "__main__":
    main()
