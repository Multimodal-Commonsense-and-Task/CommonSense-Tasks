"""
Direct (1-step) inference: send video + task query to the model without
intermediate description generation.

This is the baseline inference mode used with the standard LLaVA-OneVision
checkpoint (llava-onevision-qwen2-7b-ov).  Unlike the 2-step MeCoT
pipeline (infer_describe.py -> infer_wdesc.py), there is no video description
or reminder query — the model answers the task question directly.

Supported question types:
    - action_summary              (Task 1: action recognition)
    - action_summary_between      (Task 2: relative query)
    - action_niah_single/double/triple  (Task 3: needle-in-a-haystack)
    - action_anomaly_domain_loc   (Task 4: domain anomaly detection)
    - action_anomaly_patternbreak_loc  (Task 5: pattern-break detection)

Usage:
    python eval/infer_direct.py \
        --model-path <CHECKPOINT_PATH> \
        --video_path <VIDEO_DIR> \
        --gt_file <RAW_DATA_JSONL> \
        --output_dir <OUTPUT_DIR> \
        --vname2cls <VNAME2CLS_JSON> \
        --question_type action_summary \
        --conv-mode qwen_1_5 \
        --for_get_frames_num 32 \
        --mm_spatial_pool_stride 2 \
        --add_format_prompt \
        --num-chunks 8 --chunk-idx 0
"""

import argparse
import copy
import json
import math
import os
import string
import sys

# Add project root to path so that the bundled llava package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoConfig

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# ==================== Data I/O ====================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def load_json_data(path):
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    return load_json(path)


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# ==================== Video Loading ====================

def load_video(video_path, nframes, force_sample=True):
    if nframes == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > nframes or force_sample:
        if nframes < 3:
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, nframes + 2, dtype=int
            )[1:-1]
        else:
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, nframes, dtype=int
            )
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def equally_distribute_number(total, n):
    base_quota = total // n
    remainder = total % n
    result = [base_quota] * n
    for i in range(remainder):
        result[i] += 1
    return result


def load_synthesize_video(video_paths, nframes, force_sample=True):
    videos = []
    frame_time = []
    video_time = []
    n_clips = len(video_paths)
    if nframes % n_clips == 0:
        frames_per_clip = nframes // n_clips
        for video_path in video_paths:
            if os.path.exists(video_path):
                v, ft, vt = load_video(video_path, frames_per_clip, force_sample)
                videos.append(v)
                frame_time += ft
                video_time.append(vt)
    else:
        frames_list = equally_distribute_number(nframes, n_clips)
        for idx, video_path in enumerate(video_paths):
            if os.path.exists(video_path):
                v, ft, vt = load_video(video_path, frames_list[idx], force_sample)
                videos.append(v)
                frame_time += ft
                video_time.append(vt)
    return videos, frame_time, video_time


# ==================== Prompt Templates ====================

INFER_CFG = {
    "temperature": 0.0,
    "top_p": None,
    "max_new_tokens": 1024,
}

# -- Video description prefixes --
VIDEO_DESCRIPTION = (
    "The given video consists of {} distinct clips, "
    "each featuring a different human activity."
)
VIDEO_DESCRIPTION_WNUM = (
    "You will watch a video that is a combination of {} video clips "
    "about human activities."
)
VIDEO_DESCRIPTION_NIAH = (
    "You will watch a video that contains {} combinations of video clips "
    "about human activities, each showing different human activities."
)

# -- Task questions --
QUESTIONS = {
    "action_summary": (
        "Your task is to analyze the video and identify the actions "
        "present in each clip in sequential order."
    ),
    "action_summary_between": (
        "Your task is to identify the actions present in the video "
        "between the action of {} and {}."
    ),
    "action_niah_single": (
        "In what order is the [{}] action performed in the video?"
    ),
    "action_niah_double": (
        "In what order are the [{}] actions performed in the video?"
    ),
    "action_niah_triple": (
        "In what order are the [{}] actions performed in the video?"
    ),
    "action_anomaly_domain_loc": (
        "In these activity clips, all but one action belong to a single "
        "semantic category. At which position in the sequence does the "
        "outlier action occur?"
    ),
    "action_anomaly_patternbreak_loc": (
        "In these activity clips, there is a repeating pattern of actions, "
        "interrupted by a single anomalous action. At which position in the "
        "sequence does the pattern-breaking action occur?"
    ),
}

# -- Format post-prompts --
FORMAT_POSTPROMPT = {
    "action_summary": (
        "The response format must be a Python list containing only the "
        "corresponding action labels (e.g., ['C', 'A', 'B', 'D']). "
        "The list must have exactly {} different elements"
    ),
    "action_summary_between": (
        "Your answer must be in Python list format, containing only the "
        "corresponding alphabets (e.g., ['C', 'A', 'B']). Do not include "
        "the two mentioned actions, {} and {} in your answer list. "
        "The list must have exactly {} element(s)."
    ),
    "action_niah_single": (
        "Answer ONLY the exact one position number in the integer format "
        "ranging from 1 to {}. (e.g., [1])"
    ),
    "action_niah_double": (
        "Answer ONLY the exact two position numbers in list format "
        "as integers ranging from 1 to {}. (e.g., [1, 3])"
    ),
    "action_niah_triple": (
        "Answer ONLY the exact three position numbers in list format "
        "as integers ranging from 1 to {}. (e.g., [1, 2, 3])"
    ),
    "action_anomaly_domain_loc": (
        "Answer ONLY the exact one position number in the integer format "
        "ranging from 1 to {}. (e.g., 1)"
    ),
    "action_anomaly_patternbreak_loc": (
        "Answer ONLY the exact one position number in the integer format "
        "ranging from 1 to {}. (e.g., 1)"
    ),
}

# -- Class / action-count prompts (Task 1, 2 only) --
CLASS_PROMPT = (
    "You must use only the predefined action classes listed below, "
    "ensuring they align with the activities described in the video.\n"
    "The available action classes are:\n"
)
CLASS_PROMPT_RELATIONSHIP_DUAL = (
    "You must choose the actions from the human action classes listed "
    "below. Do not include the two mentioned actions ([{}], [{}]) in "
    "your answer.\nClasses:\n"
)
NACTION_PROMPT_STRICT = (
    "Your answer should contain exactly {} different actions "
    "in the order they appear in the video."
)


# ==================== Query Construction ====================

def _get_task_subset_value(dp, idx):
    """Look up action name from task_subset (handles both dict and list)."""
    action_key = dp["class_order"][idx]
    ts = dp.get("task_subset")
    if ts is None:
        return action_key
    if isinstance(ts, dict):
        return ts.get(str(action_key), action_key)
    return action_key


def build_query(dp, question_type, add_format_prompt=False,
                add_class_prompt=False, add_naction_helper=False):
    """Build task-specific query from raw JSONL data.

    Returns (query, answer, n_gt_classes, options).
    """
    n_clips = len(dp["class_order"])
    options = None

    # ---- Task 1: action_summary ----
    if question_type == "action_summary":
        query = VIDEO_DESCRIPTION.format(n_clips) + "\n" + QUESTIONS["action_summary"]
        answer = list(dp["class_order"])
        n_gt = len(answer)

        if add_class_prompt:
            ts = dp.get("task_subset", [])
            classes = list(ts.values()) if isinstance(ts, dict) else list(ts)
            options = [f"{string.ascii_uppercase[i]}. {c}" for i, c in enumerate(classes)]
            query += "\n" + CLASS_PROMPT + "\n".join(options)
            answer = [string.ascii_uppercase[classes.index(a)] for a in answer]
        if add_naction_helper:
            query += "\n" + NACTION_PROMPT_STRICT.format(n_clips)
        if add_format_prompt:
            query += "\n" + FORMAT_POSTPROMPT["action_summary"].format(n_clips)

        return query, answer, n_gt, options

    # ---- Task 2: action_summary_between ----
    if question_type == "action_summary_between":
        action1 = _get_task_subset_value(dp, dp["qaction1"])
        action2 = _get_task_subset_value(dp, dp["qaction2"])
        query = (
            VIDEO_DESCRIPTION.format(n_clips)
            + "\n"
            + QUESTIONS["action_summary_between"].format(action1, action2)
        )
        loc1, loc2 = dp["qaction1"], dp["qaction2"]
        answer = list(dp["class_order"][loc1 + 1 : loc2])
        n_gt = len(answer)

        if add_class_prompt:
            ts = dp.get("task_subset", [])
            classes = list(ts.values()) if isinstance(ts, dict) else list(ts)
            options = [f"{string.ascii_uppercase[i]}. {c}" for i, c in enumerate(classes)]
            query += "\n" + CLASS_PROMPT_RELATIONSHIP_DUAL.format(action1, action2) + "\n".join(options)
            # Convert answer: letter labels -> action names -> MCQ labels
            if isinstance(ts, dict):
                answer_names = [ts[a] for a in answer]
            else:
                answer_names = answer
            answer = [string.ascii_uppercase[classes.index(a)] for a in answer_names]
        if add_naction_helper:
            query += "\n" + NACTION_PROMPT_STRICT.format(n_gt)
        if add_format_prompt:
            query += "\n" + FORMAT_POSTPROMPT["action_summary_between"].format(
                str(dp["class_order"][dp["qaction1"]]),
                str(dp["class_order"][dp["qaction2"]]),
                n_gt,
            )

        return query, answer, n_gt, options

    # ---- Task 3: action_niah_* ----
    if question_type.startswith("action_niah"):
        variant = question_type.split("_")[-1]  # single / double / triple
        n_targets = {"single": 1, "double": 2, "triple": 3}[variant]

        targ_indices = [i - 1 for i in dp["niah_locs"]]  # 1-based -> 0-based
        targ_actions = [dp["class_order"][i] for i in targ_indices]
        answer = [i + 1 for i in targ_indices]  # 0-based -> 1-based

        if n_targets > 1:
            targ_str = ", ".join(targ_actions[:-1]) + ", " + targ_actions[-1]
        else:
            targ_str = targ_actions[0]

        query = (
            VIDEO_DESCRIPTION_NIAH.format(n_clips)
            + "\n"
            + QUESTIONS[question_type].format(targ_str)
        )
        n_gt = n_targets

        if add_format_prompt:
            query += "\n" + FORMAT_POSTPROMPT[question_type].format(n_clips)

        return query, answer, n_gt, None

    # ---- Task 4: action_anomaly_domain_loc ----
    if question_type == "action_anomaly_domain_loc":
        query = (
            VIDEO_DESCRIPTION_WNUM.format(n_clips)
            + "\n"
            + QUESTIONS["action_anomaly_domain_loc"]
        )
        answer = dp["answer"]
        n_gt = 1

        if add_format_prompt:
            query += "\n" + FORMAT_POSTPROMPT["action_anomaly_domain_loc"].format(n_clips)

        return query, answer, n_gt, None

    # ---- Task 5: action_anomaly_patternbreak_loc ----
    if question_type == "action_anomaly_patternbreak_loc":
        query = (
            VIDEO_DESCRIPTION_WNUM.format(n_clips)
            + "\n"
            + QUESTIONS["action_anomaly_patternbreak_loc"]
        )
        answer = dp["answer"]
        n_gt = 1

        if add_format_prompt:
            query += "\n" + FORMAT_POSTPROMPT["action_anomaly_patternbreak_loc"].format(n_clips)

        return query, answer, n_gt, None

    raise ValueError(f"Unknown question_type: {question_type}")


def make_data_to_send(dp, video_path, vid2cls, question_type,
                      add_format_prompt=False, add_class_prompt=False,
                      add_naction_helper=False):
    """Prepare inference data from raw JSONL (direct, no Step 1)."""
    query, answer, n_gt, options = build_query(
        dp, question_type, add_format_prompt, add_class_prompt, add_naction_helper
    )

    idx = question_type + "_" + str(dp["combination_id"])
    modal_paths = [
        os.path.join(video_path, vid2cls[v], v) for v in dp["videos"]
    ]

    return {
        "id": idx,
        "modal_paths": modal_paths,
        "nclips": len(dp["videos"]),
        "class_order": dp["class_order"],
        "query": query,
        "len_gt": n_gt,
        "options": options,
        "answer": answer,
        "modal_type": "VIDEO",
        "video_decode_backend": "frames",
        **INFER_CFG,
    }


# ==================== Argument Parsing ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct (1-step) inference without video descriptions"
    )
    parser.add_argument("--video_path", required=True, help="Directory containing video files")
    parser.add_argument("--gt_file", required=True, help="Raw data JSONL file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output JSONL")
    parser.add_argument("--vname2cls", required=True, help="Path to video-name-to-class JSON")
    parser.add_argument("--question_type", required=True, help="Task question type")
    parser.add_argument("--model-path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() != "false"), default=True)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--overwrite_infercfg", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--add_format_prompt", action="store_true", default=False)
    parser.add_argument("--add_class_prompt", action="store_true", default=False)
    parser.add_argument("--add_naction_helper_strict", action="store_true", default=False)
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


# ==================== Inference ====================

def run_inference(args):
    vid2cls = load_json(args.vname2cls)

    # Load model
    model_name = get_model_name_from_path(args.model_path)
    overwrite_config = {
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_newline_position": args.mm_newline_position,
    }

    cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

    if "qwen" not in args.model_path.lower():
        if "224" in cfg_pretrained.mm_vision_tower:
            least_token_number = (
                args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2
                + 1000
            )
        else:
            least_token_number = (
                args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2
                + 1000
            )
        scaling_factor = math.ceil(least_token_number / 4096)
        if scaling_factor >= 2:
            if "vicuna" in cfg_pretrained._name_or_path.lower():
                overwrite_config["rope_scaling"] = {
                    "factor": float(scaling_factor),
                    "type": "linear",
                }
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        overwrite_config=overwrite_config,
    )
    model.config.add_faster_video = False

    if args.overwrite_infercfg:
        INFER_CFG["temperature"] = args.temperature

    # Load data (raw JSONL)
    gt_questions = load_json_data(args.gt_file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = "1_1"
    if args.output_name:
        output_name = f"{output_name}_{args.output_name}"

    answers_file = os.path.join(args.output_dir, f"{output_name}.jsonl")

    # Resume support
    if os.path.exists(answers_file):
        res = load_jsonl(answers_file)
        res_idx = set([x["id"] for x in res])
        print(
            f"Resuming: loaded {len(res)}, total {len(gt_questions)}, "
            f"remaining {len(gt_questions) - len(res)}"
        )
    else:
        res_idx = set()

    with open(answers_file, "a") as ans_file:
        for sample in tqdm(gt_questions):
            sample_original = copy.deepcopy(sample)

            try:
                sample = make_data_to_send(
                    sample, args.video_path, vid2cls, args.question_type,
                    args.add_format_prompt, args.add_class_prompt,
                    args.add_naction_helper_strict,
                )

                if sample["id"] in res_idx:
                    continue

                qs = sample["query"]
                video_paths = sample["modal_paths"]
                videos, frame_time, video_time = load_synthesize_video(
                    video_paths, args.for_get_frames_num, args.force_sample
                )
                videos = [
                    image_processor.preprocess(video, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .cuda()
                    for video in videos
                ]
                video = torch.cat(videos, dim=0)
                video = [video]

                if model.config.mm_use_im_start_end:
                    qs = (
                        DEFAULT_IM_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN
                        + DEFAULT_IM_END_TOKEN
                        + "\n"
                        + qs
                    )
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                # Direct inference: no video description or reminder
                prompt = prompt + "\nAnswer: "

                input_ids = (
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )
                if tokenizer.pad_token_id is None:
                    if "qwen" in tokenizer.name_or_path.lower():
                        tokenizer.pad_token_id = 151643

                attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                stop_str = (
                    conv.sep
                    if conv.sep_style != SeparatorStyle.TWO
                    else conv.sep2
                )
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(
                    keywords, tokenizer, input_ids
                )

                with torch.inference_mode():
                    if "mistral" not in cfg_pretrained._name_or_path.lower():
                        output_ids = model.generate(
                            inputs=input_ids,
                            images=video,
                            attention_mask=attention_masks,
                            modalities=["video"],
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=1024,
                            top_p=0.1,
                            num_beams=1,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                    else:
                        output_ids = model.generate(
                            inputs=input_ids,
                            images=video,
                            attention_mask=attention_masks,
                            modalities=["video"],
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=1024,
                            top_p=0.1,
                            num_beams=1,
                            use_cache=True,
                        )

                outputs = (
                    tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    .strip()
                    .replace(".", "")
                )

                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()

                sample["model_prediction"] = {
                    "status": "success",
                    "message": outputs,
                }
                sample.update(**INFER_CFG)
                sample["original_jsonl"] = sample_original

                ans_file.write(json.dumps(sample) + "\n")
                ans_file.flush()

            except Exception as e:
                sample_id = sample.get("id", sample_original.get("id", "?"))
                print(f"ERROR: id={sample_id} {e}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
