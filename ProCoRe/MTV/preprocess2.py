from MTV.preprocess import *  # noqa: F401,F403

import json
import random


def get_format_func(cur_dataset):
    if cur_dataset == "vizwiz":
        return format_vizwiz
    if cur_dataset == "okvqa":
        return format_okvqa
    if cur_dataset == "flower":
        return format_flower
    if cur_dataset == "cub":
        return format_cub
    if cur_dataset == "dtd":
        return format_dtd
    return format_custom


def format_custom(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    if cur_item is None:
        sample = random.choice(all_data)
        data = sample if isinstance(sample, dict) else json.loads(sample)
        mode = "extract_query" if split == "test" else "extract_examples"
    else:
        data = cur_item if isinstance(cur_item, dict) else json.loads(cur_item)
        mode = "extract_query"

    image_list = data.get("image", [])
    question_id = data.get("id", 0)

    conversations = data.get("conversations", [])
    if not conversations:
        return "", [], "", question_id

    full_text = conversations[0]["value"]
    answer = conversations[1]["value"] if len(conversations) > 1 else ""

    is_many_shot = len(image_list) > 1 or full_text.count("Answer:") > 1

    if is_many_shot and mode == "extract_query":
        last_image_token_idx = full_text.rfind("<image>")
        if last_image_token_idx != -1:
            full_text = full_text[last_image_token_idx:].strip()
            trailing_image_tokens = full_text.count("<image>")
            if trailing_image_tokens > 0:
                image_list = image_list[-trailing_image_tokens:]

    if mode == "extract_examples":
        answer = ""

    return full_text, image_list, answer, question_id
