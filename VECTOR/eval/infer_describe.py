"""
Step 1: Generate video descriptions using LLaVA-OneVision model.

This script processes multi-clip videos and generates natural language descriptions
of the human activities shown in each clip. The output is used as input for
downstream tasks (e.g., anomaly detection in Step 2).

Usage:
    python eval/infer_describe.py \
        --model-path <CHECKPOINT_PATH> \
        --video_path <VIDEO_DIR> \
        --gt_file <INPUT_JSONL> \
        --output_dir <OUTPUT_DIR> \
        --vname2cls <VNAME2CLS_JSON> \
        --conv-mode qwen_1_5 \
        --for_get_frames_num 32 \
        --mm_spatial_pool_stride 2 \
        --num-chunks 8 --chunk-idx 0

Prerequisites:
    - LLaVA-OneVision dependencies (torch, transformers, decord, etc.)
"""

import argparse
import copy
import json
import math
import os
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


# ==================== Prompt Construction ====================

INFER_CFG = {
    "temperature": 0.0,
    "top_p": None,
    "max_new_tokens": 1024,
}

VIDEO_DESCRIPTION = (
    "The provided video consists of {} distinct clips, "
    "each showcasing different human activities."
)

DESCRIBE_PROMPT = (
    "Please provide a brief description of the video, only briefly describe "
    "the key human actions with {} bullet points. Each description should "
    "follow the format: 1 [description of first clip], 2 [description of "
    "second clip], and so on. Strictly limit each description to under 10 "
    "words, using only essential words to state the human action clearly."
)


def build_describe_query(dp):
    n_clips = len(dp["class_order"])
    query = VIDEO_DESCRIPTION.format(n_clips) + "\n" + DESCRIBE_PROMPT.format(n_clips)
    answer = dp["class_order"]
    return query, answer


def make_data_to_send(dp, video_path, vid2cls):
    query, answer = build_describe_query(dp)
    idx = "describe_" + str(dp["combination_id"])
    modal_paths = [os.path.join(video_path, vid2cls[v], v) for v in dp["videos"]]

    return {
        "id": idx,
        "modal_paths": modal_paths,
        "nclips": len(dp["videos"]),
        "class_order": dp["class_order"],
        "query": query,
        "len_gt": len(answer),
        "options": None,
        "answer": answer,
        "modal_type": "VIDEO",
        "video_decode_backend": "frames",
        **INFER_CFG,
    }


# ==================== Argument Parsing ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate video descriptions"
    )
    parser.add_argument("--video_path", required=True, help="Directory containing video files")
    parser.add_argument("--gt_file", required=True, help="Input JSONL with video combinations")
    parser.add_argument("--output_dir", required=True, help="Directory to save output JSONL")
    parser.add_argument("--vname2cls", required=True, help="Path to video-name-to-class JSON")
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

    if "multi-event" in args.model_path.lower():
        overwrite_config["vocab_size"] = 152064

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

    # Load data
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
            sample = make_data_to_send(sample, args.video_path, vid2cls)

            if sample["id"] in res_idx:
                continue

            qs = sample["query"]
            video_paths = sample["modal_paths"]

            try:
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
                            max_new_tokens=4098,
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
                            max_new_tokens=4098,
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
                print(f"ERROR: {video_paths} {e}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
