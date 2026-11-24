import random
import json
from collections import defaultdict
import os
from open_clip import create_model_and_transforms, get_tokenizer
import torch
from PIL import Image
import copy

# from ..utils.ICL_utils import get_task_instruction
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device).eval()
tokenizer = get_tokenizer('ViT-B-32')

img_sim = False
sim = "image"
order = "random"

def get_task_instruction(dataset):
    if dataset == 'classification':
        instr = "Choose from the following options and output ONLY the answer: [A)Flora, B)Walter, C)Trum, D)Clare]"
    elif dataset == 'clevr':
        instr = 'Look carefully at the image, count ONLY the objects that match the given property, and output ONLY the number.'
    elif dataset == 'clock':
        instr = ''
    elif dataset == 'operator_induction':
        instr = ""
    elif dataset == 'palindrome_dataset':
        instr = ""
    elif dataset == 'sudoku':
        instr = ""
    elif dataset == 'shapes_count':
        instr = ""
    elif dataset == 'vqav2':
        instr = ""
    return instr


random.seed(1)
# folder_names = ["operator_induction"]
folder_names = [ "palindrome_dataset", "shapes_count", "sudoku", "clock"]
# folder_names = ["classification", "clevr","operator_induction", "palindrome_dataset", "shapes_count", "sudoku", "vqav2", "clock"]
# folder_names = ["clock"]
# folder_names2 = ["operator_induction", "operator_induction_interleaved"]


# shots = [1, 2 ,4, 8, 16]
shots = [0]

for dataset_name in folder_names:
    print("start", dataset_name)
    shot_defaultdict = defaultdict(list)
    dataset_dir = os.path.join(".", dataset_name)
    
    with open(os.path.join(dataset_dir, "query.json")) as q_json:
        query_json = list(json.load(q_json))
    with open(os.path.join(dataset_dir, "support.json")) as s_json:
        support_json = json.load(s_json) 
    
    for i in range(len(query_json)):
        query = query_json[i]
        
        if "shapes_count" == dataset_name:
            support_pool = [data for data in support_json if data["feature"] == query["feature"]]
            supports = random.sample(support_pool, k=max(shots))
        else:
            supports = random.sample(support_json, k=max(shots))
            
        if img_sim == True:
            query_image = query["image"][0]
            query_image = f"./{query_image}"
            q_image = preprocess(Image.open(query_image).convert("RGB")).unsqueeze(0).to(device, dtype=torch.float16)
            # query_text = output_text.strip()
            # q_text = tokenizer([str(query_text)]).cuda()
            
            with torch.no_grad(), torch.autocast("cuda"):
                query_image_features = model.encode_image(q_image)
                # query_text_features = model.encode_text(q_text)
                query_image_features /= query_image_features.norm(dim=-1, keepdim=True)
                # query_text_features /= query_text_features.norm(dim=-1, keepdim=True)
            
                
        
        for shot in shots:
            
            if dataset_name in ["clock", "operator_induction"]:
                operator = query['operator']
                operator_index = ['+', '-', 'x'].index(operator)
            
            images = []
            chat_sequence = []
            
            if shot>0:
                selected_supports = supports[:shot]
                
                if img_sim == True:
                    selected_supports2 = copy.deepcopy(selected_supports)
                    support_images = []
                    for support in selected_supports:
                        support_image = support["image"][0]
                        support_image = f"./{support_image}"
                        support_image = Image.open(support_image)
                        support_image = preprocess(support_image).unsqueeze(0).cuda()
                        support_images.append(support_image)
                    support_image_features = []
                    with torch.no_grad():
                        for s_image in support_images:
                            s_image_feature = model.encode_image(s_image)
                            support_image_features.append(s_image_feature)
                        support_image_features = torch.cat(support_image_features)
                        # support_text_features = model.encode_text(supports_cand_text)
                        support_image_features /= support_image_features.norm(dim=-1, keepdim=True)
                        # support_text_features /= support_text_features.norm(dim=-1, keepdim=True)
                    image_probs = (100.0 * query_image_features.to(device, dtype=torch.float16) @ support_image_features.T.to(device, dtype=torch.float16)).softmax(dim=-1)[0]
                    
                    if sim == "image":
                        total_probs = image_probs
                        support_probs, support_indices = torch.topk(total_probs, shot)
                        if order == "ascend":
                            support_probs = support_probs.flip(dims=[0])
                            support_indices = support_indices.flip(dims=[0])
                        selected_supports = [selected_supports2[i.item()] for i in support_indices]
                
                input_text= get_task_instruction(dataset_name)  
                for support in selected_supports:
                    input_text += "\n"
                    if dataset_name == "vqav2":
                            image = support["image_id"]
                            image = f"./data/vqa/vqa/train2014/COCO_val2014_{str(image).zfill(12)}"
                            images.append(image)
                            input_text += f"<image>\n"
                    else:
                        for img_i, image in enumerate(support["image"]):
                            image = f"./data/{image}"
                            images.append(image)
                            input_text += f"<image>\n"
                    if dataset_name in ["clock", "operator_induction"]:
                        input_text += f"{support['question']}\nAnswer: {str(support['answer'][operator_index])}\n"
                    else:
                        input_text += f"{support['question']}\nAnswer: {str(support['answer'])}\n"
            else:
                input_text = get_task_instruction(dataset_name)  
                input_text += "\n"
                
            # images.extend(query["image"])
            if dataset_name == "vqav2":
                image = query["image_id"]
                # image = image.replace(f"{dataset_name}/", f"./data/{dataset_name}/")
                image = f"./data/vqa/vqa/val2014/COCO_val2014_{str(image).zfill(12)}"
                images.append(image)
                input_text += f"<image>\n" 
            else:
                for img_i, image in enumerate(query["image"]):
                    
                    # image = image.replace(f"{dataset_name}/", f"./data/{dataset_name}/")
                    image = f"./data/{image}"
                    images.append(image)
                    input_text += f"<image>\n" 
            if shot > 0:
                input_text += f"{query['question']} Learn from the demos and give only the answer.\nAnswer:"
            else:
                input_text += f"{query['question']} Give only the answer \nAnswer:"
            
            chat_sequence.append({"from": "human", "value": input_text})
            
            chat_sequence.append({"from": "gpt", "value": str(query["answer"])})
                
            new_item = {}
            new_item['id'] = i
            new_item['content'] = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                
            new_item["task"] = "0"
            new_item["image"] = images 
            new_item['conversations'] = chat_sequence
            
            shot_defaultdict[shot].append(new_item)
    for shot in shots:
        # print(shot, len(shot_defaultdict[shot]))
        if shot == 0:
            with open(f'/disk1/mjlee/LMM-ICL/datalists/{dataset_name}/dataset-{shot}.json', 'w') as fp_w:
                json.dump(shot_defaultdict[shot], fp_w)
        else:
            if order == "random":
                with open(f'/mnt/disk2/mjlee/LMM-ICL/datalists/{dataset_name}/dataset-{shot}_truemicl.json', 'w') as fp_w:
                    json.dump(shot_defaultdict[shot], fp_w)
            elif order == "descend":
                with open(f'/mnt/disk2/mjlee/LMM-ICL/datalists/{dataset_name}/dataset-{shot}_imgsim_des.json', 'w') as fp_w:
                    json.dump(shot_defaultdict[shot], fp_w)
            elif order == "ascend":
                with open(f'/mnt/disk2/mjlee/LMM-ICL/datalists/{dataset_name}/dataset-{shot}_imgsim_asc.json', 'w') as fp_w:
                    json.dump(shot_defaultdict[shot], fp_w)
        
            
        
        
        