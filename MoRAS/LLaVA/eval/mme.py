import sys
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model

from utils.functions_3stage import *
from utils.utils import *
from datasets import load_dataset


import math
import re
import shortuuid


# functions for inference
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



def main():
    args = parse_args()
    disable_torch_init()
    
    if args.eval_model == "llava_v1_5_7b":
        model_path = "liuhaotian/llava-v1.5-7b"
    elif args.eval_model == "llava_v1_5_13b":
        model_path = "liuhaotian/llava-v1.5-13b"

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        args.load_8bit, args.load_4bit, device=args.device
    )






    questions = [json.loads(q) for q in open(os.path.expanduser('../datasets/utility/MME/llava_mme.jsonl'), "r")]
    questions = get_chunk(questions, 1, 0)
    answers_file = os.path.expanduser(f'./results/{args.eval_model}/{args.mode}/mme.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
        
    
    for line in tqdm(questions):
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        
        question = cur_prompt
        img_path = os.path.join('../datasets/utility/MME/MME_Benchmark_release_version', idx)

        
        
        
        
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"


        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()


        if type(img_path) is str:
            image = load_image(img_path)
        else:
            image = img_path

        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)

        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)


        sigmoid_x, sigmoid_k = get_sigmoid_parameters(args.eval_model)
        print(sigmoid_x, sigmoid_k)
        
        rephrased_question = generate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_size, question)
        rephrased_question = get_prompt3(rephrased_question, question)

        conv = conv_templates[args.conv_mode].copy()

        response1, base_state1 = evaluate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_size, rephrased_question, first_token=False)


        shifting_coefficient = get_shifting_coefficient(args, model, base_state1, sigmoid_x, sigmoid_k)

        conv = conv_templates[args.conv_mode].copy()
        final_response = generate_final_output(args, model, conv, tokenizer, image_tensor, image_size, question, sigmoid_x, sigmoid_k, shifting_coefficient)

        print(final_response)


        final_answer = final_response
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": final_answer,
                                   "answer_id": ans_id,
                                   "model_id": 'llava',
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        
    ans_file.close()


if __name__ == "__main__":
    main()