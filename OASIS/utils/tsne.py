
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

# from utils.method_manager_VLM import select_method
# from torch.utils.tensorboard import SummaryWriter

from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num#, can_infer
from tqdm import tqdm

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava
from models.bunny import conversation as conversation_lib_bunny
from models.duallora.dualloralayer import DualLoraLayer
from models.dual_ia3.dual_ia3_layer import DualIA3Layer

import warnings
import time
import datetime
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import StoppingCriteria, StoppingCriteriaList

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

ALPHABET = ['A','B','C','D','E','F']

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

def evaluate(dataset, save_name, round, model, tokenizer, device, model_args, training_args, logger, client_id=None, batch_size=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, collate_fn=DataCollatorForGenerationDataset(tokenizer))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    
    if 'llava' in model_args.model_name_or_path.lower():
        conv = conversation_lib_llava.default_conversation
    elif 'bunny' in model_args.model_name_or_path.lower():
        conv = conversation_lib_bunny.default_conversation
    repeat_criteria = CustomStoppingCriteria()
    stop_str = conv.sep2
    keywords = [stop_str]
    
    # img_feat_size = 729
    predictions = []
    n_word_total = 0
    n_generated_word_total = 1
    n_word_correct = 1
    cnt = 0
    # for i, (inputs, imgs, golds, prompts, img_files) in enumerate(tqdm(dataloader)):
    all_hidden_states = []
    for i, batch in enumerate((dataloader)): #tqdm
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
        
        
        with torch.inference_mode():
            output = model.generate(
                inputs,
                attention_mask=attention_mask,
                images=imgs,
                do_sample=True,# if args.temperature > 0 else False,
                temperature=training_args.eval_temp,#args.temperature,
                top_p=None,#args.top_p,
                num_beams=1,#args.num_beams,
                max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                )
        reps = output.hidden_states[-1][-1][0]
        reps = torch.mean(reps, dim=0).to(torch.float)
        # print("reps shape", reps.view(-1).shape)
        all_hidden_states.append(reps.cpu())
    
    all_hidden_states = torch.stack(all_hidden_states)

    return all_hidden_states
    
def gaussian_kernel(X, Y, sigma):
    X_norm = X.pow(2).sum(1).view(-1, 1)
    Y_norm = Y.pow(2).sum(1).view(1, -1)
    pairwise_dists = X_norm + Y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))

    # pairwise_dists = cdist(X, Y, 'sqeuclidean')
    K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
    return K

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

    os.makedirs(f"eval_results/{training_args.mode}/2025_{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'eval_results/{training_args.mode}/2025_{training_args.note}/round_{training_args.round_to_eval}.log', mode="w")

    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}/federated')

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    # train_datalists, test_datalists = get_datalists(training_args, training_args.setup, training_args.scenario)
    
    batch_size = 1
    
    logger.info(f'Evaluatiing clients and server at round {training_args.round_to_eval}')
    start_time = time.time()
    server_eval_key = []
    
    with open(f"./scenarios/{training_args.setup}/scenario-{training_args.scenario}.json") as fp:
        scenario = json.load(fp)
    
    datasets = scenario[0]["datasets"]
    total_rounds = len(datasets)
    dataset_names = [dataset["dataset"] for dataset in datasets]
    print()
    experiment_name = training_args.note
    save_name = training_args.note
    
    cul_full_data = []
    cul_full_data_round = []
    cul_selected_id = []
    start_id = 0
    for round_i in range(total_rounds):
        with open(f'importance/{training_args.note}_localrank0_round{round_i}.json', 'r') as round_info:
            all_round_info = json.load(round_info)
        full_data = all_round_info["all_streamed_samples"]
        full_data_round = all_round_info["all_info_per_round"]
        selected_input_inds = all_round_info["selected_input_inds"]
        # selected_input_inds = [0,1,2]
        cul_full_data.extend(full_data)
        cul_full_data_round.extend(full_data_round)
        for ind in selected_input_inds:
            print("ind", ind)
            cul_selected_id.append(start_id+ind)
        start_id += len(full_data)
            
        
    
    logger.info(f'load {training_args.note}')
    if len(training_args.pretrain_checkpoint)>10:
        pretrained_state_dict = torch.load(f'{training_args.pretrain_checkpoint}', map_location='cpu')
        model.load_state_dict(pretrained_state_dict, strict=False)
        
    print("total_data_len", len(cul_full_data))
    del full_data
    test_datalist = cul_full_data
    dataset = GenerationDataset(test_datalist, tokenizer, data_args)
    full_data_features = evaluate(dataset, save_name, training_args.round_to_eval, model, tokenizer, device, model_args, training_args, logger, None, batch_size)
    print("total_features_len", len(full_data_features))
    '''
    # full_data_features = full_data_features.cpu()    
    # print("len(full_data_features)", len(full_data_features))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(full_data_features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                    c=cul_full_data_round, cmap='tab10', s=10, alpha=0.8)
    

    # classes = np.unique(full_data_round)
    classes = dataset_names
    legend = plt.legend(handles=scatter.legend_elements()[0],
                        labels=[str(cls) for cls in classes],
                        title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"t-SNE original")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('./tsne_plots', exist_ok=True)
    plt.savefig(f"./tsne_plots/tsne_{training_args.note}.jpg", dpi=300)

    plt.figure(figsize=(8, 6))
    
    # if sampled_version:
    for ind in cul_selected_id:
        cul_full_data_round[ind] = total_rounds+1
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
        c=cul_full_data_round, cmap='tab10', s=10, alpha=0.8)
    dataset_names.append("selected")
    # else:
     # classes = np.unique(full_data_round)
    classes = dataset_names
    legend = plt.legend(handles=scatter.legend_elements()[0],
                        labels=[str(cls) for cls in classes],
                        title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"t-SNE sampled")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('./tsne_plots', exist_ok=True)
    plt.savefig(f"./tsne_plots/tsne_sampled{training_args.note}.jpg", dpi=300)
    '''
    density_list = []
    cul_selected_samples = []
    for round_i in range(total_rounds):
        with open(f'importance/{training_args.note}_localrank0_round{round_i}.json', 'r') as round_info:
            all_round_info = json.load(round_info)
        selected_input_inds = all_round_info["selected_input_inds"]
        # selected_input_inds = [0,1,2]
        # cul_full_data.extend(full_data)
        # cul_full_data_round.extend(full_data_round)
        # for ind in selected_input_inds:
            # cul_selected_id.append(start_id+ind)
        total_selected_samples = []
        for ind in selected_input_inds:
            total_selected_samples.append(full_data_features[ind])
            cul_selected_samples.append(full_data_features[ind])
    
        total_selected_samples = torch.stack(total_selected_samples)
        i_K = gaussian_kernel(total_selected_samples, total_selected_samples, 1) # gamma=1
        density = i_K.mean()
        density_list.append(density.item())
    cul_selected_samples = torch.stack(cul_selected_samples)
    i_K = gaussian_kernel(cul_selected_samples, cul_selected_samples, 1) # gamma=1
    density = i_K.mean()
    # density_list.append(density.item())
    print("Density", density_list, "Density Avg", sum(density_list)/len(density_list), "TOTAL Density", density.item())




if __name__ == "__main__":
    main()