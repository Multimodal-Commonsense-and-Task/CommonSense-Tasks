import sys, time
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



def get_unified_examine_data(pth, img_dir_pth):
    if pth.endswith(".jsonl"):
        my_input = []
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text_content = data.get("query") if data.get("query") is not None else data.get("safe_query")
                img_name = data.get("image_id") if data.get("image_id") is not None else data.get("image_pth")
                
                my_input.append([
                    {"type": "text", "content": text_content},
                    {"type": "image", "content": os.path.join(img_dir_pth, img_name)}
                ])
        return my_input

    dataset = load_from_disk(pth)
    input_to_chameleon = []
    if not dataset:
        raise ValueError("Dataset is empty.")

    sample = dataset[0]
    if 'image_1' in sample and 'options' in sample:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for item in dataset:
            image = item['image_1'].resize((256, 256))
            query = item['question']
            options = item['options']
            if isinstance(options, str): options = ast.literal_eval(options)

            prompt_text = query.strip() + "\n"
            for i, option in enumerate(options):
                prompt_text += f"{letters[i]}. {option}\n"
            prompt_text += "Please answer directly with only the letter of the correct option."

            parts = re.split(r'<image\s*\d+>', prompt_text)
            part1 = parts[0].strip()
            part2 = parts[1].strip() if len(parts) > 1 else ''

            input_to_chameleon.append([
                {"type": "text", "content": part1},
                {"type": "image", "content": image},
                {"type": "text", "content": part2},
                {"type": "answer", "content": item['answer']}
            ])
    elif 'image' in sample and 'answer' in sample:
        for item in dataset:
            image = item['image'].resize((256, 256))
            input_to_chameleon.append([
                {"type": "text", "content": item['question']},
                {"type": "image", "content": image},
                {"type": "answer", "content": item['answer']}
            ])
    else:
        raise ValueError("Unrecognized dataset format.")
    
    return input_to_chameleon



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

    print(f"Loaded model: {model_name}")
    

    dataset_dir = "../datasets/calibration/VLSafe"
    img_dir = os.path.join(dataset_dir, "images")

    input_file = os.path.join(dataset_dir, "train", "VLSafe_harmlessnss_alignment.jsonl")
    dataset1 = get_unified_examine_data(input_file, img_dir)
    input_file = os.path.join(dataset_dir, "train", "vlsafe_convert_safe.jsonl")
    dataset2 = get_unified_examine_data(input_file, img_dir)
    
    datasets = [dataset1, dataset2]
    
    cossim_list = []
    
    np.random.seed(0)
    random_indices = np.random.choice(
        np.arange(3000),
        size=50,       
        replace=False   
    )

    for dataset in datasets:

        for i, index in enumerate(tqdm(random_indices, desc="Extracting Activations")):
            item_list = dataset[index]
            
            for block in item_list:
                if block['type'] == 'text':
                    question = block['content']
                elif block['type'] == 'image':
                    img_path = block['content']
            
            
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


            rephrased_question = generate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_size, question)
            rephrased_question = get_prompt3(rephrased_question, question)

            conv = conv_templates[args.conv_mode].copy()

            response1, base_state1 = evaluate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_size, rephrased_question, args.n_token, first_token=False)


            cossim = get_cossim(
                args,
                model,
                base_state1,
                gamma=args.gamma,
                n_token=args.n_token,
            )

            cossim_list.append(cossim)
            # print(cossim)
            # print('Running Average:', np.mean(cossim_list))
        
    sigmoid_x = float(np.mean(cossim_list))

    if not np.isfinite(sigmoid_x) or sigmoid_x >= 1.0:
        raise ValueError(f"Invalid sigmoid_x: {sigmoid_x}")

    target_risk = 0.99
    sigmoid_k = np.log(target_risk / (1.0 - target_risk)) / (1.0 - sigmoid_x)

    print(f"sigmoid_x (S_base): {sigmoid_x:.6f}")
    print(f"sigmoid_k (alpha):  {sigmoid_k:.6f}")

if __name__ == "__main__":
    main()
