import logging.config
import os
import random
import re
import string
from pathlib import Path

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel
import pandas as pd

from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, Qwen_GenerationDataset, DataCollatorForGenerationDataset, Qwen_DataCollatorForGenerationDataset, LazySupervisedDataset, OneVision_DataCollatorForGenerationDataset, OneVision_GenerationDataset, Llama_DataCollatorForGenerationDataset,  Llama_GenerationDataset, Otter_GenerationDataset, Otter_DataCollatorForGenerationDataset, OneVision1_5_GenerationDataset, OneVision1_5_DataCollatorForGenerationDataset, QwenVL3_GenerationDataset, QwenVL3_GenerationDataset_CoT # Intern_GenerationDataset, Intern_DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num, denormalize_bbox_if_needed, parse_gold_bbox, parse_bbox_from_text, calculate_iou
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

import ollama

# =============================================================================
# Prompts & constants
# =============================================================================
ALPHABET = ['A','B','C','D','E','F']
CHOICES = {"classification": ["A)Flora", "B)Walter", "C)Trum", "D)Clare"], "nlvr2":["True", "False"], "pope":["Yes", "No"], "AMBER":["yes", "no"]}
# idk_values = ["ambiguous", "bad question", "cannot confirm", "depend", "don’t know", "it is difficult", "i can’t", "none", "not clear", "not sure", "sorry", "hard to determine", "not possible", "uncertain", "unanswerable", "unknow"]
idk_values = [
    'ambiguous',
    'depend',
    "i can't",
    'not sure',
    'not possible',
    "unknown",
    "bad question",
    "don't know",
    "none",
    "sorry",
    "uncertain",
    "cannot confirm",
    "it is difficult",
    "not clear",
    "hard to determine",
    "unanswerable",
    "i am not sure",
    'it is impossible',
    'it is not possible',
    "i don't know",
    'it is unknown',
    "i'm not sure",
    'it is unclear',
    "i can't determine",
    'it is ambiguous',
    'there is no',
    'there are no',
    'is not',
    'are not',
    'unknown',
    'no specific',
]

SMMILE_PROMPT = (
    "A medical AI model is provided with an image and asked the question \"{question}\". "
    "The correct answer to this question is: \"{answer}\". The AI model outputs \"{response}\" as its " + \
    "response. Is the AI model correct? Please output your answer as a single digit, where 1 " + \
    "indicates that the AI model is correct and 0 indicates that the AI model is incorrect with " + \
    "respect to the correct answer. Do not provide anything other than the digit in your response."
)

# =============================================================================
# Generation stopping criteria
# =============================================================================
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
    
# =============================================================================
# LLM-as-judge scoring
# =============================================================================
def run_llm_judge(judge_records, judge_model="llama3.3"):
    """
    judge_records: list of dicts with keys
        - problem_id
        - question
        - true_answer
        - generated_answer

    Returns:
        llm_accuracy, llm_stats, llm_judgments
    """
    llm_judgment = []
    errors = []

    for rec in tqdm(judge_records, desc="Running LLM judge"):
        prompt = SMMILE_PROMPT.format(
            question=rec["question"],
            answer=rec["true_answer"],
            response=rec["generated_answer"],
        )

        try:
            llm_ans = ollama.chat(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}]
            ).message.content.strip()
        except Exception as e:
            llm_ans = ""

        if llm_ans in ["0", "1"]:
            llm_judgment.append(int(llm_ans))
            errors.append(0)
        else:
            llm_judgment.append(0)
            errors.append(1)

    judge_df = pd.DataFrame(judge_records).copy()
    judge_df["llm_judgment"] = np.array(llm_judgment).astype(bool)

    total_examples = len(judge_df)
    correct_examples = int(judge_df["llm_judgment"].sum())
    llm_accuracy = (correct_examples / total_examples) * 100 if total_examples > 0 else 0.0

    if "problem_id" in judge_df.columns:
        problem_accuracy = (judge_df.groupby("problem_id")["llm_judgment"].mean() * 100).to_dict()
        incorrect_examples = judge_df.loc[~judge_df["llm_judgment"], "problem_id"].tolist()
    else:
        problem_accuracy = {}
        incorrect_examples = []

    err_count = judge_df[judge_df["generated_answer"].astype(str).str.contains("ERROR", na=False)].shape[0]

    llm_stats = {
        "total_examples": total_examples,
        "llm_err_count": int(sum(errors)),
        "correct_examples": correct_examples,
        "accuracy": llm_accuracy,
        "err_count": int(err_count),
        "problem_accuracies": problem_accuracy,
        "incorrect_examples": incorrect_examples,
        "word_count": judge_df["generated_answer"].apply(lambda x: len(str(x).split())).mean() if total_examples > 0 else 0.0,
    }

    return llm_accuracy, llm_stats, llm_judgment


# =============================================================================
# Activation intervention: MTV (mean task vector) + FV (function vector)
# =============================================================================
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

def get_mtv_config(model, model_name_or_path):
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
        config['head_dim'] = getattr(llm.config.text_config, 'head_dim', None)
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
        text_config = model.config.text_config
        n_layers = text_config.num_hidden_layers
        config['n_heads'] = text_config.num_attention_heads
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

def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, mtv_config, model_args, dataset_name=None):
    layer_to_heads = {}
    for layer, head_n, _ in layer_head_token_pairs:
        layer = int(layer)
        head_n = int(head_n)
        if layer not in layer_to_heads:
            layer_to_heads[layer] = []
        layer_to_heads[layer].append(head_n)
    edit_layers = set(layer_to_heads.keys())
    use_additive_update = isinstance(dataset_name, str) and (
        "pororo" in dataset_name
        or "flintstones" in dataset_name
        or "storystream" in dataset_name
        or "story_visual_ICL" in dataset_name
    )

    layer_idx_cache = {}
    proj_weight_cache = {}
    head_idx_cache = {}
    act_cache = {}
    
    def rep_act(output, layer_name, inputs):
        if isinstance(inputs, tuple):
            inp = inputs[0]
        else:
            inp = inputs

        current_layer = layer_idx_cache.get(layer_name)
        if current_layer is None:
            parts = layer_name.split('.')
            try:
                current_layer = int(parts[mtv_config['split_idx']])
            except (IndexError, ValueError):
                for p in parts:
                    if p.isdigit():
                        current_layer = int(p)
                        break
            layer_idx_cache[layer_name] = current_layer
        
        if current_layer in edit_layers:
            original_shape = inp.shape
            
            # o_proj input width can differ from hidden_size (Qwen3-VL-4B:
            # 32 * 128 vs hidden_size 2560). Infer it from the traced tensor.
            if inp.size(-1) % mtv_config['n_heads'] != 0:
                raise RuntimeError(
                    f"[MTV] o_proj input width {inp.size(-1)} is not divisible by "
                    f"{mtv_config['n_heads']} heads at {layer_name}"
                )
            head_dim = inp.size(-1) // mtv_config['n_heads']
            new_shape = inp.size()[:-1] + (mtv_config['n_heads'], head_dim)
            inp_reshaped = inp.view(*new_shape)
            heads = layer_to_heads[current_layer]
            if heads:
                head_key = (current_layer, inp.device)
                head_idx = head_idx_cache.get(head_key)
                if head_idx is None:
                    head_idx = torch.tensor(heads, device=inp.device, dtype=torch.long)
                    head_idx_cache[head_key] = head_idx

                act_key = (current_layer, inp.device, inp.dtype)
                layer_act = act_cache.get(act_key)
                if layer_act is None:
                    def activation_for_head(head):
                        value = avg_activations[current_layer, head]
                        if value.dim() == 2 and value.size(0) == 1:
                            value = value[0]
                        if value.numel() != head_dim:
                            raise RuntimeError(
                                f"[MTV] activation width {value.numel()} does not match "
                                f"o_proj head width {head_dim} at {layer_name}"
                            )
                        return value.to(inp.device, dtype=inp.dtype)
                    layer_act = torch.stack([activation_for_head(h) for h in heads], dim=0)
                    act_cache[act_key] = layer_act

                if use_additive_update:
                    inp_reshaped[:, -1, head_idx, :] = inp_reshaped[:, -1, head_idx, :] + layer_act.unsqueeze(0)
                else:
                    inp_reshaped[:, -1, head_idx, :] = layer_act.unsqueeze(0)

            inp = inp_reshaped.view(*original_shape)
            
            out_proj = proj_weight_cache.get(layer_name)
            if out_proj is None:
                out_proj = get_module(model, layer_name).weight
                proj_weight_cache[layer_name] = out_proj
            
            new_output = torch.matmul(inp, out_proj.T)

            return new_output
        else:
            return output

    return rep_act

# =============================================================================
# Image preprocessing & intervention-context / data-loading helpers
# =============================================================================
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

def make_intervention_context(model, fv_data=None, mtv_data=None, dataset_name=None):
    """
    Returns a context manager:
      - nullcontext() if no intervention
      - TraceDict(...) if one intervention
      - nested TraceDict(...) if both (FV then MTV)
    """
    ctx = nullcontext()

    if fv_data is not None:
        fv_fn = fv_edit_o_proj_input_last_token(
            layer_head_token_pairs=fv_data["locations"],
            avg_activations=fv_data["activations"],
            model=model,
            cfg=fv_data["config"],
        )
        ctx = TraceDict(model, layers=fv_data["config"]["attn_hook_names"], edit_output=fv_fn)

    if mtv_data is not None:
        mtv_fn = last_replace_activation_w_avg(
            layer_head_token_pairs=mtv_data["locations"],
            avg_activations=mtv_data["activations"],
            model=model,
            mtv_config=mtv_data["config"],
            model_args=None,
            dataset_name=dataset_name,
        )
        if isinstance(ctx, nullcontext):
            ctx = TraceDict(model, layers=mtv_data["config"]["attn_hook_names"], edit_output=mtv_fn)
        else:
            # nest: enter FV TraceDict, then MTV TraceDict inside it
            ctx = _NestedContexts(ctx, TraceDict(model, layers=mtv_data["config"]["attn_hook_names"], edit_output=mtv_fn))

    return ctx


class _NestedContexts:
    def __init__(self, outer, inner):
        self.outer = outer
        self.inner = inner
    def __enter__(self):
        self.outer.__enter__()
        self.inner.__enter__()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.inner.__exit__(exc_type, exc, tb)
        self.outer.__exit__(exc_type, exc, tb)


def fv_edit_o_proj_input_last_token(layer_head_token_pairs, avg_activations, model, cfg):
    """
    FV: add avg head activation to LAST token at input of self_attn.o_proj.
    avg_activations: (L,H,hd) or (L,H,1,hd)
    """
    edit_layers = {x[0] for x in layer_head_token_pairs}

    def _get_act(layer, head):
        if avg_activations.dim() == 4:   # (L,H,1,hd)
            return avg_activations[layer, head, 0]
        return avg_activations[layer, head]  # (L,H,hd)

    def rep_act(output, layer_name, inputs):
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        if inp is None or inp.shape[1] == 1:
            return output

        # robust layer index parse
        parts = layer_name.split(".")
        current_layer = None
        try:
            current_layer = int(parts[cfg["split_idx"]])
        except Exception:
            for p in parts:
                if p.isdigit():
                    current_layer = int(p)
                    break
        if current_layer is None or current_layer not in edit_layers:
            return output

        B, T, D = inp.shape
        H = cfg["n_heads"]
        # The input width of o_proj is the concatenated attention-head width.
        # It is not necessarily the model hidden size: Qwen3-VL-4B, for
        # example, has hidden_size=2560 but 32 heads of width 128, so D=4096.
        if D % H != 0:
            raise RuntimeError(
                f"o_proj input width {D} is not divisible by {H} heads at {layer_name}"
            )
        hd = D // H
        inp_h = inp.view(B, T, H, hd)

        for (layer, head, _tok) in layer_head_token_pairs:
            if layer == current_layer:
                act = _get_act(layer, head).to(inp.device, dtype=inp.dtype)
                inp_h[:, -1, head, :] = inp_h[:, -1, head, :] + act

        inp_new = inp_h.view(B, T, D)
        W = get_module(model, layer_name).weight
        return torch.matmul(inp_new, W.T)

    return rep_act

def load_fv_data(model, model_args, device, fv_dir, topk=20):
    """
    FV artifacts from extract_fv():
      - mean_head_activations.pt  : (L,H,hd) or (L,H,1,hd)
      - top{K}_heads_by_mean_ie.json : list[{layer:int, head:int, ...}]
    """
    cfg = get_mtv_config(model, model_args.model_name_or_path)  # reuse config fn name; it's just hook config
    if not cfg.get("attn_hook_names"):
        raise RuntimeError(f"[FV] No attn_hook_names for {model_args.model_name_or_path}")

    mean_path = os.path.join(fv_dir, "mean_head_activations.pt")
    top_path  = os.path.join(fv_dir, f"top{topk}_heads_by_mean_ie.json")

    mean_head_acts = torch.load(mean_path, map_location="cpu").to(device)
    with open(top_path, "r") as f:
        top_heads = json.load(f)

    locations = [(int(h["layer"]), int(h["head"]), -1) for h in top_heads]  # token idx unused; we use last token
    return {"activations": mean_head_acts, "locations": locations, "config": cfg}

def load_image(image_file, input_size=448, max_num=12):
    image_path = Path(image_file)
    if not image_path.is_file() and len(image_path.parts) >= 3 and image_path.parts[0] == "data":
        # Some archives unpack with a duplicated dataset directory, e.g.
        # data/coco/coco/train2014 while the datalist stores data/coco/train2014.
        nested = image_path.parent.parent / image_path.parts[1] / Path(*image_path.parts[2:])
        if nested.is_file():
            image_path = nested
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values    
    
def get_datalists(args, scenario_num, shot, dataset_name, ordering):
    with open(f"./scenarios/{shot}/{dataset_name}/{ordering}/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)

    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for train_data in scenario:
        train_id = train_data['client_id']
        # train_datalist = []
        test_datalist = []
        eval_cnt = 0
        # train_cnt = 0
        for data in train_data['datasets']:
            with open(f"./datalists/{data['dataset']}/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "type": data['type'] if 'type' in data else 'open-ended',
                "data": datalist,
                "eval_cnt": eval_cnt})
            eval_cnt += len(datalist)
            print("total eval set len", len(datalist))
            # train_datalists[train_id] = train_datalist
        test_datalists[train_id] = test_datalist

    return "_", test_datalists

def parse_image_qa_pairs(qs):
    """Split interleaved "<image> ... Question/Answer" text into (question, answer) pairs.

    Returns (pairs, double_image). Shared by the idefics branch of the evaluate_* routines.
    """
    qs = qs.replace("\n", " ").strip()

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
        # Try to extract question and answer with flexible patterns
        # 1) With "Question:" and "Answer:"
        match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
        # 2) Fallback: look for last "Answer" keyword
        if not match:
            match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
        
        if match:
            q, a = match.groups()
        else:
            q, a = chunk, ""
        pairs.append((q.strip(), a.strip()))
    return pairs, double_image

def build_idefics_prompts(pairs, image_file, double_image):
    """Build idefics chat-template prompt dicts from parsed (question, answer) pairs."""
    prompts = []
    if isinstance(image_file, list):
        if double_image:
            for pair_i, pair in enumerate(pairs):
                image1, image2 = image_file[pair_i*2], image_file[pair_i*2+1]
                if pair_i == len(pairs) - 1:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: "}]}
                else:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: {pair[1]}"}]}
                prompts.append(user_dict)
        else:
            for image_file_i, image_path in enumerate(image_file):
                pair = pairs[image_file_i]
                if image_file_i == len(image_file) - 1:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: "}]}
                else:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text":f"User: {pair[0]} \nAssistant: {pair[1]}"}]}
                prompts.append(user_dict)
    return prompts

def load_intern_inputs(d, device):
    """Load InternVL pixel_values + per-image patch counts for one sample's image list."""
    pixel_values_list = []
    num_patches_list = []
    images = d['image']
    img_file = images[-1].split("/")[-1].split("_")[-1]
    for image in images:
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(device)
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.size(0))
    pixel_values = torch.cat(pixel_values_list, dim=0)  
    return pixel_values, num_patches_list, images, img_file

# =============================================================================
# Evaluation routines (one per task type)
# =============================================================================
def run_caprl_eval(predictions, training_args, logger):
    """Score generated captions with the CapRL / CapRL++ reward terms.

    CapRL measures a caption by its downstream usefulness: a text-only LLM must
    answer multiple-choice questions about the image from the caption alone.
    That is a different axis from BLEU/ROUGE/CIDEr, which only measure overlap
    with one reference, so it is reported alongside them rather than replacing
    them.  Enabled by --caprl_eval (default True for caprl/caprlpp runs).
    """
    if not getattr(training_args, "caprl_eval", False):
        return {}

    from utils.caprl_reward import caprl_evaluate_captions, configure_caprl_reward

    configure_caprl_reward(
        qa_source=training_args.caprl_qa_source,
        qa_model=training_args.caprl_qa_model,
        qa_cache_path=training_args.caprl_qa_cache_path,
        qa_num=training_args.caprl_qa_num,
        qa_all=training_args.caprl_qa_all,
        num_options=training_args.caprl_num_options,
        shuffle_qa=training_args.caprl_shuffle_qa,
        answer_model=training_args.caprl_answer_model,
        answer_device=training_args.caprl_answer_device,
        answer_batch_size=training_args.caprl_answer_batch_size,
        answer_temperature=training_args.caprl_answer_temperature,
        acc_scale=training_args.caprl_acc_scale,
        format_mode=training_args.caprl_format_mode,
        format_weight=training_args.caprl_format_weight,
        length_weight=training_args.caprl_length_weight,
        length_l1=training_args.caprl_length_l1,
        length_l2=training_args.caprl_length_l2,
        length_tokenizer=training_args.caprl_length_tokenizer,
        debug=training_args.caprl_debug,
    )

    try:
        caprl_scores = caprl_evaluate_captions(predictions)
    except Exception as exc:  # noqa: BLE001 - never lose a finished eval run
        logger.warning(f"[CapRL] caption scoring failed: {exc}")
        return {}

    if caprl_scores:
        logger.info(
            "Test (Server) | CapRL QA acc {caprl_qa_acc} | CapRL reward "
            "{caprl_reward} | CapRL++ reward {caprlpp_reward} | format "
            "{caprl_format} | length {caprl_length} | caption tokens "
            "{caprl_cap_tokens} |".format(**caprl_scores)
        )
    torch.cuda.empty_cache()
    return caprl_scores


def evaluate(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=1, shot_num=1, data=None, processor=None, mtv_data=None, fv_data=None):

    if "idefics" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        total = 0
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        
        for d in tqdm(data):
            cnt += 1
            
            # prompts = [f"{d['content']}\n"]
            prompts = []
            
            qs = d["conversations"][0]['value']
            gold = d["conversations"][1]['value']
            # parts = qs.split("Answer:")
            # pairs = []
            # for i in range(len(parts) - 1):
            #     question = parts[i].strip().split("\n")[-1]
            #     answer = parts[i+1].strip().split("\n")[0] 
            #     pairs.append((question, answer))
            
            pairs, double_image = parse_image_qa_pairs(qs)
            

            if "image" in d.keys():
                image_file = d["image"]
                prompts = build_idefics_prompts(pairs, image_file, double_image)
            new_image_file = [Image.open(image_path).convert("RGB") for image_path in image_file]
            prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)
            
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)
            
            if training_args.mode == "dara":
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt", do_resize=False)
                img_pos_ranges = consecutive_ranges(inputs["input_ids"][0], model.config.image_token_id)
            else:
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt")
                
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_pos_ranges = [consecutive_ranges(input_id, model.config.image_token_id) for input_id in inputs["input_ids"]]
            if training_args.mode == "dara":
                inputs['img_pos_ranges'] = img_pos_ranges
            
            with torch.inference_mode():
                with trace_context:
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)

            org_gold = d["conversations"][1]['value']
            pred_sentence = response[0].strip().lower()
            org_gold = org_gold.strip().lower()
            filter_id = "\nassistant: "
            index = pred_sentence.rfind(filter_id)
            if index!=-1:
                pred_sentence = pred_sentence[index + len(filter_id):]
            input_label = tokenizer.encode(org_gold)
            output_id = tokenizer.encode(pred_sentence)
            n_word = len(set(input_label))
            n_generated_word = len(set(output_id))
            n_correct = matching_token_num(output_id, input_label)
            
            n_word_total += n_word
            n_generated_word_total += n_generated_word
            n_word_correct += n_correct

            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold})
        
    elif "intern" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        for d in tqdm(data):
            cnt += 1
            if 'image' in d:
                pixel_values, num_patches_list, images, img_file = load_intern_inputs(d, device)
            question = d["conversations"][0]['value']
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)    
            
            with torch.inference_mode():
                with trace_context:
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list,
                        history=None, return_history=True)
                
            if len(response.strip()) == 0:
                print(f"[WARNING] Empty response generated for question")

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
            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold})
    else:
        if 'onevision-1.5' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision1_5_DataCollatorForGenerationDataset(tokenizer))
        elif 'onevision-qwen' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision_DataCollatorForGenerationDataset(tokenizer))
        elif 'Qwen' in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))
        elif 'llava' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
        elif "vl" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Llama_DataCollatorForGenerationDataset(tokenizer))
        elif 'otter' in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Otter_DataCollatorForGenerationDataset(tokenizer))

        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]
        
        # img_feat_size = 729
        predictions = []
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        cnt = 0
        correct = 0
        results_list = []
        imids = []
        for i, batch in enumerate(tqdm(dataloader)): #tqdm
            cnt+=1
            if "Qwen" in model_args.model_name_or_path:
                inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                attention_mask = batch['attention_mask'].to(device=device)[0]
            elif "onevision-1.5" in model_args.model_name_or_path.lower():
                inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                attention_mask = batch['attention_mask'].to(device=device)[0]
            elif "vl" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            elif "otter" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            else:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            inputs = inputs.to(device=device, non_blocking=True)

            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])

            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)
            with torch.inference_mode():
                with trace_context:
                    if "Qwen" in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                        )
                    elif "onevision-1.5" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        )
                    elif "onevision" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            image_sizes=image_sizes,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        )
                    elif 'llava' in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    elif 'otter' in model_args.model_name_or_path.lower():
                        # if imgs is not None and imgs.ndim == 5:
                        #     imgs = imgs.unsqueeze(2)
                        output_ids = model.generate(
                            vision_x=imgs,
                            lang_x=inputs,
                            attention_mask=attention_mask,
                            do_sample=training_args.mode!="dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                        )
                    else:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    
            input_token_len = inputs.shape[1]
            output_ids = output_ids[:,input_token_len:]

            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
            # breakpoint()
            if "Qwen" in model_args.model_name_or_path:
                filter_id = "\nassistant\n"
                for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                    pred_sentence = pred_sentence.strip().lower()
                    org_gold = gold.lower()
                    index = pred_sentence.find(filter_id)
                    if index!=-1:
                        pred_sentence = pred_sentence[index + len(filter_id):]
                    input_label = tokenizer.encode(gold)
                    output_id = tokenizer.encode(pred_sentence)
                    n_word = len(set(input_label))
                    n_generated_word = len(set(output_id))
                    n_correct = matching_token_num(output_id, input_label)
                    predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip()})
                    n_word_total += n_word
                    n_generated_word_total += n_generated_word
                    n_word_correct += n_correct
            else:
                if 'otter' not in model_args.model_name_or_path.lower():
                    filter_id = "assistant\n"
                for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                    if 'otter' not in model_args.model_name_or_path.lower():
                        index = pred_sentence.find(filter_id)
                        if index != -1:
                            pred_sentence = pred_sentence[index + len(filter_id):]
                    pred_sentence = pred_sentence.strip().lower()
                    gold = gold.lower()
                    input_label = tokenizer.encode(gold)
                    output_id = tokenizer.encode(pred_sentence)
                    n_word = len(set(input_label))
                    n_generated_word = len(set(output_id))
                    n_correct = matching_token_num(output_id, input_label)
                    predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip()})
                    
                    n_word_total += n_word
                    n_generated_word_total += n_generated_word
                    n_word_correct += n_correct
                
        
            if "coco" in dataname:
                pred_val = {}
                match = re.search(r"(\d+)\.jpg$", img_files[0][-1])
                if match:
                    number = int(match.group(1))
                    pred_val["image_id"] = number
                    pred_val["caption"] = pred_sentence
                    results_list.append(pred_val)
            
                imids.append(number)

    scores = NLPEvaluator(predictions).evaluate()
    scores["precision"] = n_word_correct / n_word_total
    scores["recall"] = n_word_correct / n_generated_word_total

    caprl_scores = run_caprl_eval(predictions, training_args, logger)
    if caprl_scores:
        scores.update(caprl_scores)

    predictions.append(scores)
    
    logger.info(f"Test (Server) | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
    
    result_filename = "results.json"
    output_path = f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}/{result_filename}"

    with open(output_path, 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()
    print("saved at", output_path)
    
def evaluate_llmjudge(
    dataset,
    dataname,
    round,
    model,
    tokenizer,
    device,
    model_args,
    training_args,
    logger,
    batch_size=1,
    shot_num=1,
    data=None,
    processor=None,
    mtv_data=None,
    fv_data=None,
    judge_model="llama3.3",
):
    predictions = []
    judge_records = []
    if "idefics" in model_args.model_name_or_path.lower():
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding

        for sample_idx, d in enumerate(tqdm(data)):
            prompts = []

            qs = d["conversations"][0]["value"]
            gold = d["conversations"][1]["value"]
            original_question_for_judge = qs

            qs = qs.replace("\n", " ").strip()

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
                match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                if not match:
                    match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)

                if match:
                    q, a = match.groups()
                else:
                    q, a = chunk, ""
                pairs.append((q.strip(), a.strip()))

            image_file = d.get("image", None)
            if image_file is None:
                continue

            if isinstance(image_file, list):
                if double_image:
                    for pair_i, pair in enumerate(pairs):
                        if pair_i == len(pairs) - 1:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: "},
                                ],
                            }
                        else:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"},
                                ],
                            }
                        prompts.append(user_dict)
                else:
                    for image_file_i, image_path in enumerate(image_file):
                        pair = pairs[image_file_i]
                        if image_file_i == len(image_file) - 1:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: "},
                                ],
                            }
                        else:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"},
                                ],
                            }
                        prompts.append(user_dict)

            new_image_file = [Image.open(image_path).convert("RGB") for image_path in image_file]
            prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)

            trace_context = make_intervention_context(
                model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname
            )

            if training_args.mode == "dara":
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt", do_resize=False)
                img_pos_ranges = [consecutive_ranges(input_id, model.config.image_token_id) for input_id in inputs["input_ids"]]
                inputs["img_pos_ranges"] = img_pos_ranges
            else:
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                with trace_context:
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)

            org_gold = d["conversations"][1]["value"].strip().lower()
            pred_sentence = response[0].strip().lower()

            filter_id = "\nassistant: "
            index = pred_sentence.rfind(filter_id)
            if index != -1:
                pred_sentence = pred_sentence[index + len(filter_id):]

            problem_id = d.get("problem_id", str(sample_idx))

            predictions.append(
                {
                    "problem_id": problem_id,
                    "sentence": pred_sentence,
                    "gt_sentence": org_gold,
                    "question": original_question_for_judge,
                }
            )
            judge_records.append(
                {
                    "problem_id": problem_id,
                    "question": original_question_for_judge,
                    "true_answer": org_gold,
                    "generated_answer": pred_sentence,
                }
            )

    elif "intern" in model_args.model_name_or_path.lower():
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode != "dara")

        for sample_idx, d in enumerate(tqdm(data)):
            pixel_values = None
            num_patches_list = None

            if "image" in d:
                pixel_values_list = []
                num_patches_list = []
                images = d["image"]
                for image in images:
                    pv = load_image(image, max_num=12).to(torch.bfloat16).to(device)
                    pixel_values_list.append(pv)
                    num_patches_list.append(pv.size(0))
                pixel_values = torch.cat(pixel_values_list, dim=0)

            question = d["conversations"][0]["value"]

            trace_context = make_intervention_context(
                model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname
            )

            with torch.inference_mode():
                with trace_context:
                    response, history = model.chat(
                        tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True,
                    )

            pred_sentence = response.strip().lower()
            org_gold = d["conversations"][1]["value"].strip().lower()
            problem_id = d.get("problem_id", str(sample_idx))

            predictions.append(
                {
                    "problem_id": problem_id,
                    "sentence": pred_sentence,
                    "gt_sentence": org_gold,
                    "question": question,
                }
            )
            judge_records.append(
                {
                    "problem_id": problem_id,
                    "question": question,
                    "true_answer": org_gold,
                    "generated_answer": pred_sentence,
                }
            )

    else:
        if "onevision-1.5" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0,
                drop_last=False, collate_fn=OneVision1_5_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "onevision-qwen" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0,
                drop_last=False, collate_fn=OneVision_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "Qwen" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0,
                drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "llava" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0,
                drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer)
            )
        elif "vl" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0,
                drop_last=False, collate_fn=Llama_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "otter" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0,
                drop_last=False, collate_fn=Otter_DataCollatorForGenerationDataset(tokenizer)
            )
        else:
            raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]

        for i, batch in enumerate(tqdm(dataloader)):
            if "Qwen" in model_args.model_name_or_path:
                inputs, imgs, golds, img_files, prompts, image_grid_thw = (
                    batch["input_ids"], batch["pixel_values"], batch["gold"],
                    batch["image_file"], batch["prompt"], batch["image_grid_thw"]
                )
                attention_mask = batch["attention_mask"].to(device=device)[0]
            elif "onevision-1.5" in model_args.model_name_or_path.lower():
                inputs, imgs, golds, img_files, prompts, image_grid_thw = (
                    batch["input_ids"], batch["pixel_values"], batch["gold"],
                    batch["image_file"], batch["prompt"], batch["image_grid_thw"]
                )
                attention_mask = batch["attention_mask"].to(device=device)[0]
            elif "vl" in model_args.model_name_or_path or "otter" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"], batch["pixel_values"], batch["gold"],
                    batch["prompt"], batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)
            else:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"], batch["images"], batch["gold"],
                    batch["prompt"], batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)

            inputs = inputs.to(device=device, non_blocking=True)

            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                    image_sizes = [x.shape[-2:] for x in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    image_sizes = None
            else:
                image_sizes = None

            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])

            trace_context = make_intervention_context(
                model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname
            )

            with torch.inference_mode():
                with trace_context:
                    if "Qwen" in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )
                    elif "onevision-1.5" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            max_new_tokens=model_args.max_new_tokens,
                        )
                    elif "onevision" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            image_sizes=image_sizes,
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            max_new_tokens=model_args.max_new_tokens,
                        )
                    elif "llava" in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )
                    elif "otter" in model_args.model_name_or_path.lower():
                        input_token_len = inputs.shape[1]
                        output_ids = model.generate(
                            vision_x=imgs,
                            lang_x=inputs,
                            attention_mask=attention_mask,
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )
                        output_ids = output_ids[:, input_token_len:]
                    else:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )

            input_token_len = inputs.shape[1]
            output_ids = output_ids[:, input_token_len:]
            pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            if "Qwen" in model_args.model_name_or_path:
                filter_id = "\nassistant\n"
            elif "otter" not in model_args.model_name_or_path.lower():
                filter_id = "assistant\n"
            else:
                filter_id = None

            batch_problem_ids = batch.get("problem_id", None)

            for batch_idx, (pred_sentence, gold, img_file) in enumerate(zip(pred_sentences, golds, img_files)):
                if filter_id is not None:
                    index = pred_sentence.find(filter_id)
                    if index != -1:
                        pred_sentence = pred_sentence[index + len(filter_id):]

                pred_sentence = pred_sentence.strip().lower()
                gold = gold.strip().lower()

                q_text = prompts[batch_idx] if isinstance(prompts, list) else str(prompts)

                if batch_problem_ids is None:
                    pid = f"{i}_{batch_idx}"
                elif isinstance(batch_problem_ids, list):
                    pid = batch_problem_ids[batch_idx]
                else:
                    pid = f"{i}_{batch_idx}"

                predictions.append(
                    {
                        "problem_id": pid,
                        "image_file": img_file,
                        "sentence": pred_sentence,
                        "gt_sentence": gold,
                        "question": q_text,
                    }
                )
                judge_records.append(
                    {
                        "problem_id": pid,
                        "question": q_text,
                        "true_answer": gold,
                        "generated_answer": pred_sentence,
                    }
                )
    llm_accuracy, llm_stats, llm_judgments = run_llm_judge(
        judge_records,
        judge_model=judge_model,
    )

    for idx in range(len(predictions)):
        predictions[idx]["llm_judgment"] = int(llm_judgments[idx])

    summary = {
        "llm_accuracy": llm_accuracy,
        "llm_correct_examples": llm_stats["correct_examples"],
        "llm_err_count": llm_stats["llm_err_count"],
        "llm_problem_accuracies": llm_stats["problem_accuracies"],
        "llm_incorrect_examples": llm_stats["incorrect_examples"],
        "avg_generated_word_count": llm_stats["word_count"],
        "total_examples": llm_stats["total_examples"],
    }


    predictions.append(summary)

    result_filename = "results.json"
    output_path = (
        f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/"
        f"{training_args.dataset_name}/scenario{training_args.scenario}/"
        f"{model_args.model_name}/{result_filename}"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(predictions, fp, indent=4)

    logger.info(
        f"Test (LLM Judge) | "
        f"ACC {summary['llm_accuracy']:.4f} | "
        f"Correct {summary['llm_correct_examples']} | "
        f"JudgeErr {summary['llm_err_count']} | "
        f"Total {summary['total_examples']} |"
    )

    torch.cuda.empty_cache()
    print("saved at", output_path)

    return summary

def evaluate_exactmatch(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=1, shot_num=0, data=None, processor=None, mtv_data=None, fv_data=None):
    _p40_shard = os.environ.get("P40_SHARD")  # [p40] "i/n": this process evaluates samples i, i+n, i+2n, ...; shards are merged by plot_tools/merge_p40_shards.py
    if _p40_shard:
        _p40_si, _p40_sn = map(int, _p40_shard.split("/"))
        if isinstance(data, list):
            data = data[_p40_si::_p40_sn]
        if dataset is not None and hasattr(dataset, "__len__"):
            dataset = torch.utils.data.Subset(dataset, list(range(_p40_si, len(dataset), _p40_sn)))
        print(f"[p40] shard {_p40_si}/{_p40_sn}: {len(data) if isinstance(data, list) else (len(dataset) if dataset is not None else '?')} samples")
    if "idefics" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        total = 0
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        
        for d in tqdm(data):
            cnt += 1
            
            prompts = []
            
            qs = d["conversations"][0]['value']
            gold = d["conversations"][1]['value']

            pairs, double_image = parse_image_qa_pairs(qs)

            if "image" in d.keys():
                image_file = d["image"]
                prompts = build_idefics_prompts(pairs, image_file, double_image)
            new_image_file = [Image.open(image_path).convert("RGB") for image_path in image_file]
            prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)
            

            if training_args.mode == "dara":
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt", do_resize=False)
                img_pos_ranges = consecutive_ranges(inputs["input_ids"][0], model.config.image_token_id)
            else:
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt")
                
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if training_args.mode == "dara":
                inputs['img_pos_ranges'] = img_pos_ranges
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)

            with torch.inference_mode():
                with trace_context:
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            org_gold = d["conversations"][1]['value']
            pred_sentence = response[0].strip().lower()
            org_gold = org_gold.strip().lower()
            filter_id = "\nassistant: "
            index = pred_sentence.rfind(filter_id)
            if index!=-1:
                pred_sentence = pred_sentence[index + len(filter_id):]
            
            # text = '>what text is shown in the red box?\nthe text \"stop\" is shown in the red box.'
            # find the text inside escaped quotes
            
            match = re.search(r'\d+', pred_sentence)
            if match:
                prediction = match.group()
                if str(prediction).lower() == org_gold:
                    acc.append(1)
                    result = "correct"
                else:
                    result = "wrong"
                    acc.append(0) 
            elif re.findall(r'"(.*?)"', pred_sentence):
                prediction = re.findall(r'"(.*?)"', pred_sentence)[0]
                if str(prediction).lower() == org_gold:
                    acc.append(1)
                    result = "correct"
                else:
                    result = "wrong"
                    acc.append(0) 
            elif "answer" in pred_sentence:
                pred = pred_sentence.split("answer: ")[-1]
                if "answer.\n" in pred:
                    pred = pred.split("answer.\n")[-1]
                elif "answer: " in pred:
                    pred = pred.split("answer: ")[-1]
                elif "answer\n" in pred:
                    pred = pred.split("answer\n")[-1]
                
                if "." in pred:
                    pred = pred.replace(".", "")
                # if result_i["gt_sentence"] == pred:
                if org_gold == pred:
                    result = "correct"
                    acc.append(1)
                else:
                    result = "wrong"
                    acc.append(0) 
            else:
                prediction = ''
                if "." in pred_sentence:
                    pred_sentence = pred_sentence.replace(".", "")
                if org_gold == pred_sentence:
                    acc.append(1)
                    result = "correct"
                else:
                    acc.append(0)
                    result = "wrong"
            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold.strip()})

    elif "intern" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        for d in tqdm(data):
            cnt += 1
            if 'image' in d:
                pixel_values, num_patches_list, images, img_file = load_intern_inputs(d, device)
            question = d["conversations"][0]['value']
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)

            with torch.inference_mode():
                with trace_context:
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list,
                        history=None, return_history=True)
                    
            org_gold = d["conversations"][1]['value']
            pred_sentence = response.strip().lower()
            org_gold = org_gold.strip().lower()
            print("pred", pred_sentence)
            match = re.search(r'\d+', pred_sentence)
            if match:
                prediction = match.group()
                if str(prediction).lower() == org_gold:
                    acc.append(1)
                    result = "correct"
                else:
                    result = "wrong"
                    acc.append(0) 
            else:
                prediction = ''
                if org_gold == pred_sentence:
                    acc.append(1)
                    result = "correct"
                else:
                    acc.append(0)
                    result = "wrong"
            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold.strip()})
    else:
        if 'onevision-1.5' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision1_5_DataCollatorForGenerationDataset(tokenizer))
        elif 'onevision-qwen' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision_DataCollatorForGenerationDataset(tokenizer))
        elif 'Qwen' in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))
        elif 'llava' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
        elif "vl" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Llama_DataCollatorForGenerationDataset(tokenizer))
        elif 'otter' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Otter_DataCollatorForGenerationDataset(tokenizer))
        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]
        
        predictions = []
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        cnt = 0
        acc = []
        
        for i, batch in enumerate(tqdm(dataloader)): #tqdm
            if "Qwen" in model_args.model_name_or_path:
                inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                attention_mask = batch['attention_mask'].to(device=device)[0]
            elif "onevision-1.5" in model_args.model_name_or_path.lower():
                inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                attention_mask = batch['attention_mask'].to(device=device)[0]
            elif "vl" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            elif "otter" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            else:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            
            inputs = inputs.to(device=device, non_blocking=True)
            
            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)
            
            with torch.inference_mode():
                with trace_context:
                    if "Qwen" in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    elif "onevision-1.5" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        )
                    elif "onevision" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs[0],
                            image_sizes=image_sizes,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        )
                    elif 'llava' in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    elif 'otter' in model_args.model_name_or_path.lower():
                        # if imgs is not None and imgs.ndim == 5:
                        #     imgs = imgs.unsqueeze(2)
                        with torch.inference_mode():
                            input_token_len = inputs.shape[1]
                            output_ids = model.generate(
                                vision_x=imgs,
                                lang_x=inputs,
                                attention_mask=attention_mask,
                                do_sample=training_args.mode!="dara",
                                temperature=training_args.eval_temp,
                                top_p=None,
                                num_beams=1,
                                max_new_tokens=model_args.max_new_tokens,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria = stopping_criteria,
                            )
                    else:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    if "vl" in model_args.model_name_or_path:
                        input_token_len = inputs.shape[1]
                        output_ids = output_ids[:,input_token_len:]
                    pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
                    # breakpoint()
                    if "Qwen" in model_args.model_name_or_path:
                        filter_id = "\nassistant"
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            index = pred_sentence.find(filter_id)
                            if index != -1:
                                pred_sentence = pred_sentence[index + len(filter_id):]
                            pred_sentence = pred_sentence.strip().lower()
                            
                            gold = gold.lower()
                            match = re.search(r'\d+', pred_sentence)
                            if match:
                                prediction = match.group()
                                if str(prediction).lower() == gold:
                                    acc.append(1)
                                    result = "correct"
                                else:
                                    result = "wrong"
                                    acc.append(0) 
                            else:
                                prediction = ''
                                if gold == pred_sentence:
                                    acc.append(1)
                                    result = "correct"
                                else:
                                    acc.append(0)
                                    result = "wrong"
                            predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip(), "result": result})
                    elif 'otter' in model_args.model_name_or_path.lower():
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            pred_sentence_raw = pred_sentence.strip(".").lower()
                            gold = gold.lower()
                            
                            prediction_to_save = pred_sentence_raw
                            match = re.search(r'\d+', pred_sentence_raw)
                            if match:
                                prediction = match.group()
                                prediction_to_save = prediction
                                if str(prediction).lower() == gold:
                                    acc.append(1)
                                    result = "correct"
                                else:
                                    result = "wrong"
                                    acc.append(0) 
                            else:
                                prediction = ''
                                if gold == pred_sentence_raw:
                                    acc.append(1)
                                    result = "correct"
                                else:
                                    acc.append(0)
                                    result = "wrong"
                            predictions.append({"image_file":img_file, "sentence":prediction_to_save, "gt_sentence":gold.strip(), "result": result})
                    else:
                        filter_id = "assistant\n"
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                    
                            index = pred_sentence.find(filter_id)
                            if index != -1:
                                pred_sentence = pred_sentence[index + len(filter_id):]
                            pred_sentence = pred_sentence.strip().lower()
                            gold = gold.lower()
                            
                            match = re.search(r'\d+', pred_sentence)
                            if match:
                                prediction = match.group()
                                if str(prediction).lower() == gold:
                                    acc.append(1)
                                    result = "correct"
                                else:
                                    result = "wrong"
                                    acc.append(0) 
                            else:
                                prediction = ''
                                if gold == pred_sentence:
                                    acc.append(1)
                                    result = "correct"
                                else:
                                    acc.append(0)
                                    result = "wrong"
                            predictions.append({"image_file":img_file, "sentence":pred_sentence, "gt_sentence":gold.strip(), "result": result})
                
    avg_acc = np.average(acc)
    scores = {}
    scores["accuracy"] = avg_acc
    
    predictions.append(scores)
    #save predictions
    logger.info(f"Test (Server) | accuracy {scores['accuracy']} |")

    result_filename = "results.json" if not _p40_shard else f"results_shard{_p40_si}of{_p40_sn}.json"
    output_path = f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}/{result_filename}"

    with open(output_path, 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()
    print("saved at", output_path)

def evaluate_grounding(
    dataset,
    dataname,
    round,
    model,
    tokenizer,
    device,
    model_args,
    training_args,
    logger,
    batch_size=1,
    shot_num=0,
    data=None,
    processor=None,
    mtv_data=None,
    fv_data=None
):
    if "idefics" in model_args.model_name_or_path.lower():
        predictions = []
        acc = []
        iou_list = []

        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode != "dara")

        for d in tqdm(data):
            prompts = []

            qs = d["conversations"][0]["value"]
            gold = d["conversations"][1]["value"]
            qs = qs.replace("\n", " ").strip()

            if "<image> <image>" in qs:
                double_image = True
                chunks = [c.strip() for c in qs.split("<image> <image>") if c.strip()]
                if not qs.startswith("<image> <image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]
            else:
                double_image = False
                chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
                if not qs.startswith("<image>"):
                    chunks = [" ".join(chunks[:2])] + chunks[2:]

            pairs = []
            for chunk in chunks:
                match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
                if not match:
                    match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)

                if match:
                    q, a = match.groups()
                else:
                    q, a = chunk, ""
                pairs.append((q.strip(), a.strip()))

            image_file = d["image"]
            if isinstance(image_file, list):
                if double_image:
                    for pair_i, pair in enumerate(pairs):
                        if pair_i == len(pairs) - 1:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: "}
                                ]
                            }
                        else:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"}
                                ]
                            }
                        prompts.append(user_dict)
                else:
                    for image_file_i, _ in enumerate(image_file):
                        pair = pairs[image_file_i]
                        if image_file_i == len(image_file) - 1:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: "}
                                ]
                            }
                        else:
                            user_dict = {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"}
                                ]
                            }
                        prompts.append(user_dict)

            new_image_file = [Image.open(image_path).convert("RGB") for image_path in image_file]
            prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)

            if training_args.mode == "dara":
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt", do_resize=False)
                img_pos_ranges = consecutive_ranges(inputs["input_ids"][0], model.config.image_token_id)
            else:
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}

            if training_args.mode == "dara":
                inputs["img_pos_ranges"] = img_pos_ranges

            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)

            with torch.inference_mode():
                with trace_context:
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)

            pred_sentence = response[0].strip().lower()
            gold_bbox = parse_gold_bbox(gold)
            pred_bbox = parse_bbox_from_text(pred_sentence)

            image_for_eval = image_file[-1] if isinstance(image_file, list) else image_file
            gold_bbox_px = denormalize_bbox_if_needed(gold_bbox, image_file=image_for_eval)
            pred_bbox_px = denormalize_bbox_if_needed(pred_bbox, image_file=image_for_eval)

            if pred_bbox_px is None or gold_bbox_px is None or len(pred_bbox_px) != 4 or len(gold_bbox_px) != 4:
                iou = 0.0
                result = "wrong"
                acc.append(0)
            else:
                iou = calculate_iou(pred_bbox_px, gold_bbox_px)
                result = "correct" if iou > 0.5 else "wrong"
                acc.append(1 if iou > 0.5 else 0)

            iou_list.append(iou)
            predictions.append({
                "image_file": image_for_eval,
                "sentence": pred_sentence,
                "pred_bbox": pred_bbox,
                "gt_bbox": gold_bbox,
                "iou": float(iou),
                "result": result
            })

    elif "intern" in model_args.model_name_or_path.lower():
        predictions = []
        acc = []
        iou_list = []

        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode != "dara")

        for d in tqdm(data):
            if "image" in d:
                pixel_values_list = []
                num_patches_list = []
                images = d["image"]
                for image in images:
                    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(device)
                    pixel_values_list.append(pixel_values)
                    num_patches_list.append(pixel_values.size(0))
                pixel_values = torch.cat(pixel_values_list, dim=0)

            question = d["conversations"][0]["value"]

            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)

            with torch.inference_mode():
                with trace_context:
                    response, history = model.chat(
                        tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=True
                    )

            pred_sentence = response.strip().lower()
            gold = d["conversations"][1]["value"]

            gold_bbox = parse_gold_bbox(gold)
            pred_bbox = parse_bbox_from_text(pred_sentence)

            image_for_eval = d["image"][-1] if isinstance(d["image"], list) else d["image"]
            gold_bbox_px = denormalize_bbox_if_needed(gold_bbox, image_file=image_for_eval)
            pred_bbox_px = denormalize_bbox_if_needed(pred_bbox, image_file=image_for_eval)

            if pred_bbox_px is None or gold_bbox_px is None or len(pred_bbox_px) != 4 or len(gold_bbox_px) != 4:
                iou = 0.0
                result = "wrong"
                acc.append(0)
            else:
                iou = calculate_iou(pred_bbox_px, gold_bbox_px)
                result = "correct" if iou > 0.5 else "wrong"
                acc.append(1 if iou > 0.5 else 0)

            iou_list.append(iou)
            predictions.append({
                "image_file": image_for_eval,
                "sentence": pred_sentence,
                "pred_bbox": pred_bbox,
                "gt_bbox": gold_bbox,
                "iou": float(iou),
                "result": result
            })

    else:
        if "onevision-1.5" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False,
                collate_fn=OneVision1_5_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "onevision-qwen" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False,
                collate_fn=OneVision_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "Qwen" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False,
                collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "llava" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False,
                collate_fn=DataCollatorForGenerationDataset(tokenizer)
            )
        elif "vl" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False,
                collate_fn=Llama_DataCollatorForGenerationDataset(tokenizer)
            )
        elif "otter" in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False,
                collate_fn=Otter_DataCollatorForGenerationDataset(tokenizer)
            )

        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]

        predictions = []
        acc = []
        iou_list = []

        for i, batch in enumerate(tqdm(dataloader)):
            if "Qwen" in model_args.model_name_or_path:
                inputs, imgs, golds, img_files, prompts, image_grid_thw = (
                    batch["input_ids"],
                    batch["pixel_values"],
                    batch["gold"],
                    batch["image_file"],
                    batch["prompt"],
                    batch["image_grid_thw"]
                )
                attention_mask = batch["attention_mask"].to(device=device)[0]
            elif "onevision-1.5" in model_args.model_name_or_path.lower():
                inputs, imgs, golds, img_files, prompts, image_grid_thw = (
                    batch["input_ids"],
                    batch["pixel_values"],
                    batch["gold"],
                    batch["image_file"],
                    batch["prompt"],
                    batch["image_grid_thw"]
                )
                attention_mask = batch["attention_mask"].to(device=device)[0]
            elif "vl" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"],
                    batch["pixel_values"],
                    batch["gold"],
                    batch["prompt"],
                    batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)
            elif "otter" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"],
                    batch["pixel_values"],
                    batch["gold"],
                    batch["prompt"],
                    batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)
            else:
                inputs, imgs, golds, prompts, img_files = (
                    batch["input_ids"],
                    batch["images"],
                    batch["gold"],
                    batch["prompt"],
                    batch["image_file"]
                )
                attention_mask = batch["attention_mask"].to(device=device)

            inputs = inputs.to(device=device, non_blocking=True)

            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs] if isinstance(imgs, list) else None

            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])

            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)

            with torch.inference_mode():
                with trace_context:
                    if "Qwen" in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )
                    elif "onevision-1.5" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            max_new_tokens=model_args.max_new_tokens,
                        )
                    elif "onevision" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs[0],
                            image_sizes=image_sizes,
                            do_sample=training_args.mode != "dara",
                            max_new_tokens=model_args.max_new_tokens,
                        )
                    elif "llava" in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )
                    elif "otter" in model_args.model_name_or_path.lower():
                        input_token_len = inputs.shape[1]
                        output_ids = model.generate(
                            vision_x=imgs,
                            lang_x=inputs,
                            attention_mask=attention_mask,
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )
                        output_ids = output_ids[:, input_token_len:]
                    else:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            do_sample=training_args.mode != "dara",
                            temperature=training_args.eval_temp,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=model_args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=stopping_criteria,
                        )

                    if "vl" in model_args.model_name_or_path:
                        input_token_len = inputs.shape[1]
                        output_ids = output_ids[:, input_token_len:]

                    pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                    if "Qwen" in model_args.model_name_or_path:
                        filter_id = "\nassistant"
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            index = pred_sentence.find(filter_id)
                            if index != -1:
                                pred_sentence = pred_sentence[index + len(filter_id):]
                            pred_sentence = pred_sentence.strip().lower()

                            gold_bbox = parse_gold_bbox(gold)
                            eval_img_file = img_file[-1] if isinstance(img_file, list) else img_file
                            pred_bbox = parse_bbox_from_text(pred_sentence)

                            gold_bbox_px = denormalize_bbox_if_needed(gold_bbox, image_file=eval_img_file)
                            pred_bbox_px = denormalize_bbox_if_needed(pred_bbox, image_file=eval_img_file)

                            if pred_bbox_px is None or gold_bbox_px is None:
                                iou = 0.0
                                result = "wrong"
                                acc.append(0)
                            else:
                                iou = calculate_iou(pred_bbox_px, gold_bbox_px)
                                result = "correct" if iou > 0.5 else "wrong"
                                acc.append(1 if iou > 0.5 else 0)
                            iou_list.append(iou)
                            predictions.append({
                                "image_file": eval_img_file,
                                "sentence": pred_sentence,
                                "pred_bbox": pred_bbox,
                                "gt_bbox": gold_bbox,
                                "iou": float(iou),
                                "result": result
                            })

                    elif "otter" in model_args.model_name_or_path.lower():
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            pred_sentence = pred_sentence.strip().lower()
                            gold_bbox = parse_gold_bbox(gold)
                            eval_img_file = img_file[-1] if isinstance(img_file, list) else img_file
                            pred_bbox = parse_bbox_from_text(pred_sentence)

                            gold_bbox_px = denormalize_bbox_if_needed(gold_bbox, image_file=eval_img_file)
                            pred_bbox_px = denormalize_bbox_if_needed(pred_bbox, image_file=eval_img_file)

                            if pred_bbox_px is None or gold_bbox_px is None:
                                iou = 0.0
                                result = "wrong"
                                acc.append(0)
                            else:
                                iou = calculate_iou(pred_bbox_px, gold_bbox_px)
                                result = "correct" if iou > 0.5 else "wrong"
                                acc.append(1 if iou > 0.5 else 0)

                            iou_list.append(iou)
                            predictions.append({
                                "image_file": eval_img_file,
                                "sentence": pred_sentence,
                                "pred_bbox": pred_bbox,
                                "gt_bbox": gold_bbox,
                                "iou": float(iou),
                                "result": result
                            })
                    else:
                        filter_id = "assistant\n"
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            index = pred_sentence.find(filter_id)
                            if index != -1:
                                pred_sentence = pred_sentence[index + len(filter_id):]
                            pred_sentence = pred_sentence.strip().lower()

                            gold_bbox = parse_gold_bbox(gold)
                            eval_img_file = img_file[-1] if isinstance(img_file, list) else img_file
                            pred_bbox = parse_bbox_from_text(pred_sentence)

                            gold_bbox_px = denormalize_bbox_if_needed(gold_bbox, image_file=eval_img_file)
                            pred_bbox_px = denormalize_bbox_if_needed(pred_bbox, image_file=eval_img_file)
                            
                            if pred_bbox_px is None or gold_bbox_px is None:
                                iou = 0.0
                                result = "wrong"
                                acc.append(0)
                            else:
                                iou = calculate_iou(pred_bbox_px, gold_bbox_px)
                                result = "correct" if iou > 0.5 else "wrong"
                                acc.append(1 if iou > 0.5 else 0)
                        
                            iou_list.append(iou)
                            predictions.append({
                                "image_file": eval_img_file,
                                "sentence": pred_sentence,
                                "pred_bbox": pred_bbox,
                                "gt_bbox": gold_bbox,
                                "iou": float(iou),
                                "result": result
                            })

    avg_acc = float(np.average(acc)) if len(acc) > 0 else 0.0
    mean_iou = float(np.average(iou_list)) if len(iou_list) > 0 else 0.0

    scores = {
        "accuracy": avg_acc,
        "mean_iou": mean_iou
    }
    predictions.append(scores)

    logger.info(f"Test (Server) | grounding accuracy {scores['accuracy']} | mean_iou {scores['mean_iou']} |")

    result_filename = "results.json"
    output_path = f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}/{result_filename}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(predictions, fp, indent=4)

    torch.cuda.empty_cache()
    print("saved at", output_path)
    

def evaluate_inclusion(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=1, shot_num=0, data=None, processor=None, mtv_data=None, fv_data=None):
    if "idefics" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        total = 0
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        
        for d in tqdm(data):
            cnt += 1
            
            prompts = []
            
            qs = d["conversations"][0]['value']
            gold = d["conversations"][1]['value']

            pairs, double_image = parse_image_qa_pairs(qs)

            if "image" in d.keys():
                image_file = d["image"]
                prompts = build_idefics_prompts(pairs, image_file, double_image)
            new_image_file = [Image.open(image_path).convert("RGB") for image_path in image_file]
            prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)

            if training_args.mode == "dara":
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt", do_resize=False)
                img_pos_ranges = consecutive_ranges(inputs["input_ids"][0], model.config.image_token_id)
            else:
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt")
                
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if training_args.mode == "dara":
                inputs['img_pos_ranges'] = img_pos_ranges
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)

            with torch.inference_mode():
                with trace_context:
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            pred_sentence = response[0].strip().lower()
            filter_id = "\nassistant: "
            index = pred_sentence.rfind(filter_id)
            if index!=-1:
                pred_sentence = pred_sentence[index + len(filter_id):]
            
            org_gold = idk_values
            
            # text = '>what text is shown in the red box?\nthe text \"stop\" is shown in the red box.'
            # find the text inside escaped quotes
            
            if any(sub in pred_sentence.lower() for sub in org_gold):
                acc.append(1)
                result = "correct"
            else:
                result = "wrong"
                acc.append(0) 
            
            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold, "result": result})

    elif "intern" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        for d in tqdm(data):
            cnt += 1
            if 'image' in d:
                pixel_values, num_patches_list, images, img_file = load_intern_inputs(d, device)
            question = d["conversations"][0]['value']
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)

            with torch.inference_mode():
                with trace_context:
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list,
                        history=None, return_history=True)
                    
            org_gold = idk_values
            pred_sentence = response.strip().lower()
            filter_id = "\nassistant: "
            index = pred_sentence.rfind(filter_id)
            if index!=-1:
                pred_sentence = pred_sentence[index + len(filter_id):]
            # text = '>what text is shown in the red box?\nthe text \"stop\" is shown in the red box.'
            # find the text inside escaped quotes
            if any(sub in pred_sentence.lower() for sub in org_gold):
                acc.append(1)
                result = "correct"
            else:
                result = "wrong"
                acc.append(0) 
            
            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold, "result": result})
    else:
        if 'onevision-1.5' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision1_5_DataCollatorForGenerationDataset(tokenizer))
        elif 'onevision-qwen' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision_DataCollatorForGenerationDataset(tokenizer))
        elif 'Qwen' in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))
        elif 'llava' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
        elif "vl" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Llama_DataCollatorForGenerationDataset(tokenizer))
        elif 'otter' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Otter_DataCollatorForGenerationDataset(tokenizer))
        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]
        
        predictions = []
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        cnt = 0
        acc = []
        
        for i, batch in enumerate(tqdm(dataloader)): #tqdm
            if "Qwen" in model_args.model_name_or_path:
                inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                attention_mask = batch['attention_mask'].to(device=device)[0]
            elif "onevision-1.5" in model_args.model_name_or_path.lower():
                inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                attention_mask = batch['attention_mask'].to(device=device)[0]
            elif "vl" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            elif "otter" in model_args.model_name_or_path:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            else:
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=device)
            
            inputs = inputs.to(device=device, non_blocking=True)
            
            if imgs is not None:
                if isinstance(imgs, list):
                    imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                else:
                    imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                image_sizes = [x.shape[-2:] for x in imgs]
            keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
            stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)
            
            with torch.inference_mode():
                with trace_context:
                    if "Qwen" in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    elif "onevision-1.5" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs[0],
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            image_grid_thw=image_grid_thw[0].to(device),
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        )
                    elif "onevision" in model_args.model_name_or_path.lower():
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs[0],
                            image_sizes=image_sizes,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                        )
                    elif 'llava' in model_args.model_name_or_path:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            images=imgs,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    elif 'otter' in model_args.model_name_or_path.lower():
                        # if imgs is not None and imgs.ndim == 5:
                        #     imgs = imgs.unsqueeze(2)
                        with torch.inference_mode():
                            input_token_len = inputs.shape[1]
                            output_ids = model.generate(
                                vision_x=imgs,
                                lang_x=inputs,
                                attention_mask=attention_mask,
                                do_sample=training_args.mode!="dara",
                                temperature=training_args.eval_temp,
                                top_p=None,
                                num_beams=1,
                                max_new_tokens=model_args.max_new_tokens,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria = stopping_criteria,
                            )
                    else:
                        output_ids = model.generate(
                            inputs,
                            attention_mask=attention_mask,
                            pixel_values=imgs,
                            do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                            temperature=training_args.eval_temp,#args.temperature,
                            top_p=None,#args.top_p,
                            num_beams=1,#args.num_beams,
                            max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria = stopping_criteria,
                            # prompt=prompts if training_args.is_prompt else None,
                        )
                    if "vl" in model_args.model_name_or_path:
                        input_token_len = inputs.shape[1]
                        output_ids = output_ids[:,input_token_len:]
                    pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
                    # breakpoint()
                    filter_id = "\nassistant"
                    if "Qwen" in model_args.model_name_or_path:
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            
                            pred_sentence = pred_sentence.strip().lower()
                            org_gold = idk_values
                            index = pred_sentence.rfind(filter_id)
                            if index != -1:
                                pred_sentence = pred_sentence[index + len(filter_id):]
                            
                            # text = '>what text is shown in the red box?\nthe text \"stop\" is shown in the red box.'
                            # find the text inside escaped quotes
                            
                            if any(sub in pred_sentence.lower() for sub in org_gold):
                                acc.append(1)
                                result = "correct"
                            else:
                                result = "wrong"
                                acc.append(0) 
                            print({"sentence":pred_sentence, "gt_sentence":org_gold, "result": result})
                            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold, "result": result})
                    elif 'otter' in model_args.model_name_or_path.lower():
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            pred_sentence_raw = pred_sentence.strip(".").lower()
                            gold = gold.lower()
                            
                            pred_sentence = pred_sentence_raw
                            org_gold = idk_values
                            index = pred_sentence.rfind(filter_id)
                            if index != -1:
                                pred_sentence = pred_sentence[index + len(filter_id):]
                            
                            # text = '>what text is shown in the red box?\nthe text \"stop\" is shown in the red box.'
                            # find the text inside escaped quotes
                            
                            if any(sub in pred_sentence.lower() for sub in org_gold):
                                acc.append(1)
                                result = "correct"
                            else:
                                result = "wrong"
                                acc.append(0) 
                            
                            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold, "result": result})
                    else:
                        for pred_sentence, gold, img_file in zip(pred_sentences, golds, img_files):
                            
                            index = pred_sentence.find(filter_id)
                            if index != -1:
                                pred_sentence = pred_sentence[index + len(filter_id):]
                            pred_sentence = pred_sentence.strip().lower()
                            
                            org_gold = idk_values
                            
                            # text = '>what text is shown in the red box?\nthe text \"stop\" is shown in the red box.'
                            # find the text inside escaped quotes
                            
                            if any(sub in pred_sentence.lower() for sub in org_gold):
                                acc.append(1)
                                result = "correct"
                            else:
                                result = "wrong"
                                acc.append(0) 
                            
                            predictions.append({"sentence":pred_sentence, "gt_sentence":org_gold, "result": result})
                
    avg_acc = np.average(acc)
    scores = {}
    scores["accuracy"] = avg_acc
    
    predictions.append(scores)
    #save predictions
    logger.info(f"Test (Server) | accuracy {scores['accuracy']} |")

    result_filename = "results.json"
    output_path = f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}/{result_filename}"

    with open(output_path, 'w') as fp:
        json.dump(predictions, fp, indent=4)
    torch.cuda.empty_cache()
    print("saved at", output_path)

def evaluate_choices(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, batch_size=2, shot_num=0, data=None, processor=None, mtv_data=None, fv_data=None):
    
    if "idefics" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        total = 0
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        
        for d in tqdm(data):
            cnt += 1
            
            # prompts = [f"{d['content']}\n"]
            prompts = []
            
            qs = d["conversations"][0]['value']
            gold = d["conversations"][1]['value']
            
            pairs, double_image = parse_image_qa_pairs(qs)

            if "image" in d.keys():
                image_file = d["image"]
                prompts = build_idefics_prompts(pairs, image_file, double_image)
            new_image_file = [Image.open(image_path).convert("RGB") for image_path in image_file]
            prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)
            processor.image_processor.do_image_splitting = False
            if training_args.mode == "dara":
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt", do_resize=False)
                img_pos_ranges = consecutive_ranges(inputs["input_ids"][0], model.config.image_token_id)
            else:
                inputs = processor(text=prompt, images=new_image_file, return_tensors="pt")
                
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if training_args.mode == "dara":
                inputs['img_pos_ranges'] = img_pos_ranges
            
            # inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)
            
            with torch.inference_mode():
                # response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list,
                #     history=None, return_history=True)
                with trace_context:
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
            org_gold = d["conversations"][1]['value']
            pred_sentence = response[0].strip().lower()
            org_gold = org_gold.strip().lower()
            filter_id = "\nassistant: "
            # index = pred_sentence.find(filter_id)
            index = pred_sentence.rfind(filter_id)
            if index!=-1:
                pred_sentence = pred_sentence[index + len(filter_id):]
            if "classification" in dataname:
                choices = CHOICES["classification"]
            elif "nlvr2" in dataname:
                choices = CHOICES["nlvr2"]
            elif "pope" in dataname:
                choices = CHOICES["pope"]
            elif "AMBER" in dataname:
                choices = CHOICES["AMBER"]
            else:
                choices = parse_choice_list(d["conversations"][0]['value'])
            
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
            predictions.append({"image_file":image_file, "sentence":pred_sentence, "gt_sentence":org_gold.strip(), 'status':status})
        
    elif "intern" in model_args.model_name_or_path.lower():
        predictions = []
        correct = 0
        cnt = 0
        
        total = 0
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        
        model = (model.eval() if len(set(getattr(model, "hf_device_map", {}).values())) > 1 else model.eval().to(device))  # [p40] keep device_map sharding
        generation_config = dict(max_new_tokens=1024, do_sample=training_args.mode!="dara")
        results_list = []
        imids = []
        acc = []
        for d in tqdm(data):
            cnt += 1
            if 'image' in d:
                pixel_values, num_patches_list, images, img_file = load_intern_inputs(d, device)
            question = d["conversations"][0]['value']
            
            # trace_context = nullcontext()
            # if mtv_data is not None:
            #     intervention_fn = last_replace_activation_w_avg(
            #         layer_head_token_pairs=mtv_data['locations'],
            #         avg_activations=mtv_data['activations'],
            #         model=model,
            #         mtv_config=mtv_data['config'],
            #         model_args=model_args
            #     )
            #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
            trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)
            
            with torch.inference_mode():
                with trace_context:
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list,
                        history=None, return_history=True)

            org_gold = d["conversations"][1]['value']
            pred_sentence = response.strip().lower()
            org_gold = org_gold.strip().lower()
            if "classification" in dataname:
                choices = CHOICES["classification"]
            elif "nlvr2" in dataname:
                choices = CHOICES["nlvr2"]
            elif "pope" in dataname:
                choices = CHOICES["pope"]
            elif "AMBER" in dataname:
                choices = CHOICES["AMBER"]
            else:
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
        if 'onevision-1.5' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision1_5_DataCollatorForGenerationDataset(tokenizer))
        elif 'onevision-qwen' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=OneVision_DataCollatorForGenerationDataset(tokenizer))
        elif 'llava' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
        elif 'Qwen' in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Qwen_DataCollatorForGenerationDataset(tokenizer))
        elif "vl" in model_args.model_name_or_path:
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Llama_DataCollatorForGenerationDataset(tokenizer))
        elif 'otter' in model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=Otter_DataCollatorForGenerationDataset(tokenizer))

        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]
        
        model.eval()
        predictions = []
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)): #tqdm
                if "Qwen" in model_args.model_name_or_path:
                    inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                    attention_mask = batch['attention_mask'].to(device=device)[0]
                elif "onevision-1.5" in model_args.model_name_or_path.lower():
                    inputs, imgs, golds, img_files, prompts, image_grid_thw = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['image_file'], batch['prompt'], batch['image_grid_thw']
                    attention_mask = batch['attention_mask'].to(device=device)[0]
                elif "vl" in model_args.model_name_or_path:
                    inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                    attention_mask = batch['attention_mask'].to(device=device)
                elif 'otter' in model_args.model_name_or_path.lower():
                    inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['pixel_values'], batch['gold'], batch['prompt'], batch['image_file']
                    attention_mask = batch['attention_mask'].to(device=device)
                else:
                    inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
                    attention_mask = batch['attention_mask'].to(device=device)
                    
                inputs = inputs.to(device=device, non_blocking=True)

                if imgs is not None:
                    if isinstance(imgs, list):
                        imgs = [img.to(device=device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                    else:
                        imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    image_sizes = [x.shape[-2:] for x in imgs]
                keyword_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs)
                stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
                # trace_context = nullcontext()
                # if mtv_data is not None:
                #     intervention_fn = last_replace_activation_w_avg(
                #         layer_head_token_pairs=mtv_data['locations'],
                #         avg_activations=mtv_data['activations'],
                #         model=model,
                #         mtv_config=mtv_data['config'],
                #         model_args=model_args
                #     )
                #     trace_context = TraceDict(model, layers=mtv_data['config']['attn_hook_names'], edit_output=intervention_fn)
                trace_context = make_intervention_context(model, fv_data=fv_data, mtv_data=mtv_data, dataset_name=dataname)
                
                with torch.inference_mode():
                    with trace_context:
                        if "Qwen" in model_args.model_name_or_path:
                            output_ids = model.generate(
                                inputs[0],
                                attention_mask=attention_mask,
                                pixel_values=imgs,
                                image_grid_thw=image_grid_thw[0].to(device),
                                # image_sizes=image_sizes,
                                do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                                temperature=training_args.eval_temp,#args.temperature,
                                # top_p=None,#args.top_p,
                                # num_beams=1,#args.num_beams,
                                max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                                # use_cache=True,
                                # pad_token_id=tokenizer.eos_token_id,
                                # stopping_criteria = stopping_criteria,
                            )
                        
                        elif "onevision-1.5" in model_args.model_name_or_path.lower():
                            output_ids = model.generate(
                                inputs[0],
                                attention_mask=attention_mask,
                                pixel_values=imgs,
                                image_grid_thw=image_grid_thw[0].to(device),
                                do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                                temperature=training_args.eval_temp,#args.temperature,
                                max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                            )
                        elif "onevision" in model_args.model_name_or_path.lower():
                            output_ids = model.generate(
                                inputs,
                                attention_mask=attention_mask,
                                images=imgs,
                                image_sizes=image_sizes,
                                do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                                temperature=training_args.eval_temp,#args.temperature,
                                top_p=None,#args.top_p,
                                num_beams=1,#args.num_beams,
                                max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria = stopping_criteria,
                                # prompt=prompts if training_args.is_prompt else None,
                            )
                        elif 'llava' in model_args.model_name_or_path:
                            output_ids = model.generate(
                                inputs,
                                attention_mask=attention_mask,
                                images=imgs,
                                do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                                temperature=training_args.eval_temp,#args.temperature,
                                top_p=None,#args.top_p,
                                num_beams=1,#args.num_beams,
                                max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria = stopping_criteria,
                                # prompt=prompts if training_args.is_prompt else None,
                            )
                        elif 'otter' in model_args.model_name_or_path.lower():
                            # if imgs is not None and imgs.ndim == 5:
                            #     imgs = imgs.unsqueeze(2)
                            input_token_len = inputs.shape[1]
                            output_ids = model.generate(
                                vision_x=imgs,
                                lang_x=inputs,
                                attention_mask=attention_mask,
                                do_sample=training_args.mode!="dara",
                                temperature=training_args.eval_temp,
                                top_p=None,
                                num_beams=1,
                                max_new_tokens=model_args.max_new_tokens,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria = stopping_criteria,
                            )
                        else:
                            output_ids = model.generate(
                                inputs,
                                attention_mask=attention_mask,
                                pixel_values=imgs,
                                do_sample=training_args.mode!="dara",# if args.temperature > 0 else False,
                                temperature=training_args.eval_temp,#args.temperature,
                                top_p=None,#args.top_p,
                                num_beams=1,#args.num_beams,
                                max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                                stopping_criteria = stopping_criteria,
                                # prompt=prompts if training_args.is_prompt else None,
                            )
                if "vl" in model_args.model_name_or_path:
                    input_token_len = inputs.shape[1]
                    output_ids = output_ids[:,input_token_len:]

                pred_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
                del output_ids
                if "Qwen" in model_args.model_name_or_path:
                    filter_id = "\nassistant\n"
                    for pred_sentence, prompt, gold, img_file in zip(pred_sentences, prompts, golds, img_files):
                        pred_sentence = pred_sentence.strip()
                        
                        index = pred_sentence.find(filter_id)
                        if index!=-1:
                            pred_sentence = pred_sentence[index + len(filter_id):]
                        if "classification" in dataname:
                            choices = CHOICES["classification"]
                        elif "nlvr2" in dataname:
                            choices = CHOICES["nlvr2"]
                        elif "pope" in dataname:
                            choices = CHOICES["pope"]
                        elif "AMBER" in dataname:
                            choices = CHOICES["AMBER"]
                        else:
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
                elif 'otter' in model_args.model_name_or_path.lower():
                    for pred_sentence, prompt, gold, img_file in zip(pred_sentences, prompts, golds, img_files):
                        pred_sentence_raw = pred_sentence.strip().lower()
                        gold = gold.lower()
                            
                        if "classification" in dataname:
                            choices = CHOICES["classification"]
                        elif "nlvr2" in dataname:
                            choices = CHOICES["nlvr2"]
                        elif "pope" in dataname:
                            choices = CHOICES["pope"]
                        elif "AMBER" in dataname:
                            choices = CHOICES["AMBER"]
                        else:
                            choices = parse_choice_list(prompt)

                        pred_option = can_infer(pred_sentence_raw, choices)
                        
                        sentence_to_save = ""
                        if isinstance(pred_option, str):
                            sentence_to_save = pred_option.lower()
                            if gold.lower() == pred_option.lower():
                                correct += 1
                                status='correct'
                            else:
                                status='wrong'
                        else:
                            status = 'unkown'
                            sentence_to_save = pred_sentence_raw
                        
                        total += 1
                        predictions.append({"image_file":img_file, "sentence":sentence_to_save, "gt_sentence":gold.strip(), 'status':status})
                else:
                    filter_id = "assistant\n"
                    for pred_sentence, prompt, gold, img_file in zip(pred_sentences, prompts, golds, img_files):
                        index = pred_sentence.find(filter_id)
                        if index != -1:
                            pred_sentence = pred_sentence[index + len(filter_id):]
                        pred_sentence = pred_sentence.strip().lower()
                        gold = gold.lower()
                            
                        if "classification" in dataname:
                            choices = CHOICES["classification"]
                        elif "nlvr2" in dataname:
                            choices = CHOICES["nlvr2"]
                        elif "pope" in dataname:
                            choices = CHOICES["pope"]
                        elif "AMBER" in dataname:
                            choices = CHOICES["AMBER"]
                        else:
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
    logger.info(f"Test (Server) | accuracy {scores['accuracy']} |")

    result_filename = "results.json"
    output_path = f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}/{result_filename}"

    with open(output_path, 'w') as fp:
        json.dump(predictions, fp, indent=4)
        
    torch.cuda.empty_cache()

# =============================================================================
# Multiple-choice answer parsing
# =============================================================================
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
    
    match = re.findall(
        r'^([A-D])[\:\.]\s*(.+)$',
        input_string,
        re.MULTILINE
    )
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
    
    return []

def can_infer(answer, choices):
    answer = str(answer).lower()
    
    # Special case for ['Positive', 'Negative']
    if set(choices) == {'Positive', 'Negative'}:
        if 'yes' in answer or 'Yes' in answer:
            return 'Positive'
        elif 'no' in answer or 'No' in answer:
            return 'Negative'
    
    # Special case for ['Positive', 'Negative']
    if set(choices) == {'Yes', 'No'}:
        if 'yes' in answer or 'Yes' in answer:
            return 'Yes'
        elif 'no' in answer or 'No' in answer:
            return 'No'
    
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

# =============================================================================
# Entry point
# =============================================================================
def main():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if os.environ.get("P40_DEVICE_MAP") == "auto":  # [p40] split the model over the visible GPUs (very long prompts)
        bnb_model_from_pretrained_args["device_map"] = "auto"
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
    fileHandler = logging.FileHandler(f'./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}/round_{training_args.round_to_eval}{"_shard" + os.environ["P40_SHARD"].replace("/", "of") if os.environ.get("P40_SHARD") else ""}.log', mode="w")
    
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
    
    
    print("bring model")
    processor = None
    if "Qwen" in model_args.model_name_or_path:
        model, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
        tokenizer = processor.tokenizer
    elif "Intern" in model_args.model_name_or_path:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif "onevision" in model_args.model_name_or_path.lower():

        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif "vl" in model_args.model_name_or_path:
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif 'idefics' in model_args.model_name_or_path.lower():
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    else:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    print("brought model")

    mtv_data = None
    if hasattr(model_args, 'use_mtv') and model_args.use_mtv:
        logger.info(f"[MTV] Loading MTV from {model_args.mtv_path} and {model_args.mtv_activation_path}")
        
        try:
            activations = torch.load(model_args.mtv_activation_path, map_location=device)
            
            locations = torch.load(model_args.mtv_path, map_location=device)
            
            config = get_mtv_config(model, model_args.model_name_or_path)
            
            if not config.get('attn_hook_names'):
                raise RuntimeError(
                    f"No MTV hook config found for {model_args.model_name_or_path}"
                )
            mtv_data = {
                "activations": activations,
                "locations": locations,
                "config": config
            }
            logger.info(f"[MTV] Loaded successfully. Intervening on {len(locations)} heads.")
                
        except Exception as e:
            raise RuntimeError(
                "MTV was requested, but its intervention artifacts could not be loaded"
            ) from e
            
    
    fv_data = None
    if getattr(training_args, "mode", "") == "fv":
        # fv_dir = (
        #     f"./fv/{training_args.ordering}/{training_args.eval_shot}/"
        #     f"{training_args.dataset_name}/scenario{training_args.scenario}/"
        #     f"{model_args.model_name[:-3]}"
        # )
        try:
            fv_dir = (
                f"./fv/{training_args.ordering}/2-shot/"
                f"{training_args.dataset_name}/scenario{training_args.scenario}/"
                f"{model_args.model_name[:-3]}"
            )
        except:
            fv_dir = (
            f"./fv/{training_args.ordering}/1-shot/"
            f"{training_args.dataset_name}/scenario{training_args.scenario}/"
            f"{model_args.model_name[:-3]}"
        )
        logger.info(f"[FV] Loading FV from: {fv_dir}")
        fv_data = load_fv_data(model, model_args, device, fv_dir, topk=getattr(training_args, "fv_topk", 20))
        logger.info(f"[FV] Loaded FV heads: {len(fv_data['locations'])}")
                
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario, shot=training_args.eval_shot, dataset_name=training_args.dataset_name, ordering=training_args.ordering)
    batch_size = 1
    logger.info(f'Evaluatiing server at round {training_args.round_to_eval}')
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
            if round_i + 1 > training_args.round_to_eval:
                continue
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
            if data_info['type'] == 'open-ended':
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger,  batch_size, shot_num, data_info['data'], processor=processor, mtv_data=mtv_data, fv_data=fv_data)
            elif data_info['type'] == 'multi-choice':
                evaluate_choices(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, shot_num, data_info['data'], processor=processor, mtv_data=mtv_data, fv_data=fv_data)
            elif data_info['type'] == 'exact-match':
                evaluate_exactmatch(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, shot_num, data_info['data'], processor=processor, mtv_data=mtv_data, fv_data=fv_data)
            elif data_info['type'] == 'inclusion':
                evaluate_inclusion(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, shot_num, data_info['data'], processor=processor, mtv_data=mtv_data, fv_data=fv_data)
            elif data_info['type'] == 'grounding':
                evaluate_grounding(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, shot_num, data_info['data'], processor=processor, mtv_data=mtv_data, fv_data=fv_data)
            elif data_info['type'] == "llm-as-judge":
                evaluate_llmjudge(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, shot_num, data_info['data'], processor=processor, mtv_data=mtv_data, fv_data=fv_data)
            else:
                evaluate(dataset, data_info['data_name'], training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, batch_size, shot_num, data_info['data'], processor=processor, mtv_data=mtv_data, fv_data=fv_data)
        return
    
    logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    
if __name__ == "__main__":
    main()
