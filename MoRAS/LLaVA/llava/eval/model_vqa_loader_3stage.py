import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math



# added
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
import sys
from utils.utils import *


def sigmoid(x, x0, k=1, sigmoid_inverse=False):
    if sigmoid_inverse:
        return 1.0 / (1.0 + np.exp(-k * (-x + x0)))
    else:
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def get_kl_div(logits_p, logits_q):
    log_p = torch.nn.functional.log_softmax(logits_p, dim=1)
    q = torch.nn.functional.softmax(logits_q, dim=1)

    # Compute KL divergence: KL(p || q)
    kl = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
    return kl

def get_cross_entropy(logits_p, logits_q):
    p = F.softmax(logits_p, dim=1)
    log_q = F.log_softmax(logits_q, dim=1)
    ce = -(p * log_q).sum(dim=1)

    return ce[0]




def generate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_sizes, instruction):

    inp = get_prompt1(instruction)
    
    


    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            )
        

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs




    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)


    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            # temperature=args.temperature,
            # max_new_tokens=args.max_new_tokens,
            use_cache=False,
            max_new_tokens=1024,
            )


    generated_ids = output

    outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return outputs


def evaluate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_sizes, instruction, first_token=True, n_token=1):

    inp = get_prompt2(instruction)
    
    
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp


    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=n_token,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            )
        

    output_ids = output.sequences
    hidden_states = output.hidden_states


    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    token_states = []
    for i in range(len(hidden_states)):
        temp = []
        for j in range(len(hidden_states[i])):
            temp.append(hidden_states[i][j][0, -1].detach().cpu().numpy())
        token_states.append(temp)

    token_states = np.array(token_states)

    if first_token:
        return outputs, token_states[0]
    else:
        return outputs, token_states






    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            # temperature=args.temperature,
            # max_new_tokens=args.max_new_tokens,
            max_new_tokens=3,
            use_cache=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            )


    generated_ids = output.sequences
    hidden_states = output.hidden_states

    outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    token_states = []
    for i in range(len(hidden_states)):
        temp = []
        for j in range(len(hidden_states[i])):
            temp.append(hidden_states[i][j][0, -1].detach().cpu().numpy())
        token_states.append(temp)

    token_states = np.array(token_states)

    if first_token:
        return outputs, token_states[0]
    else:
        return outputs, token_states













def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]


        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return qs, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    qs, image_tensors, image_sizes = zip(*batch)
    # input_ids, image_tensors, image_sizes = zip(*batch)
    # input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return qs, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (qs, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):

        qs = qs[0]


        # Stage 1: Generate the rephrased question
        conv = conv_templates[args.conv_mode].copy()
        rephrased_question = generate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_sizes, qs)

        rephrased_question = get_prompt3(rephrased_question, qs)
        print('Rephrased question:')
        print(rephrased_question)


        conv = conv_templates[args.conv_mode].copy()
        response1, base_state1 = evaluate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_sizes, rephrased_question, first_token=False)


        shifting_coefficient = get_shifting_coefficient(model, base_state1, sigmoid_x, sigmoid_k)


        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0)


        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                shifting_params=[sigmoid_x, sigmoid_k, shifting_coefficient, input_ids.shape[1]+575]
                )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print('Final output:')
        print(outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        
        break
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
