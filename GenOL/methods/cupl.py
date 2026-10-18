# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.zero_shot_clip import ZeroShotClip
import torch.nn.functional as F
logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")
from torchvision import datasets
from utils.data_loader import ImageTextDataset
import os
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Subset
import json


class CuPL(ZeroShotClip):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        # if kwargs["temp_batchsize"] is None:
        #     kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        self.get_LLM_prompts()
        super().__init__(train_datalist, test_datalist, device, **kwargs)
    
    def get_LLM_prompts(self):
        with open(f"/home/vision/mjlee/New_Nameonly_CL/DomainNet_ours_prompt.json") as fp:
            self.dict = json.load(fp)
    
    def pre_tokenize(self, dict):
        zeroshot_weights=[]
        for cla in self.exposed_classes:
            # print("keys", list(self.dict.keys()))
            # print("class", cla)
            cls_prompts = self.dict[cla]
            cls_prompts = list(np.random.choice(cls_prompts, size=5))
            # print(cla, cls_prompts)
            with torch.no_grad():
                texts = self.tokenizer(cls_prompts).cuda()
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cpu()
                
    
    def online_evaluate(self, domain_name, cur_task, test_list, sample_num, batch_size, n_worker, train_list=None):
        # llm_prompts = self.get_LLM_prompts()
        # print(type(self.dict))
        
        self.pre_tokenize(self.dict)
        
        self.cur_task = cur_task
        
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        
        
        
        test_dataset = ImageTextDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            # prompt_cls_list=self.text_class_prompts
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        
        self.report_test(domain_name, sample_num, eval_dict["avg_acc"])
            
        del test_loader
        return sample_num, eval_dict


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        total_TR1, total_TR5 = 0.0, 0.0
        total_IR1, total_IR5 = 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        list_img_features = []
        list_text_features = []
        gt_labels = []
        
        

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                imgs = data["image"]
                labels = data["label"]
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # print("labels", labels)
                
                image_features = self.model.encode_image(imgs)
                # text_features = self.model.encode_text(self.text_class_tokens)
                
                list_img_features.append(image_features.cpu())
                gt_labels.append(labels.cpu())
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features /= text_features.norm(dim=-1, keepdim=True)
        
        gt_labels = torch.cat(gt_labels)
        image_features = torch.cat(list_img_features)
        # text_features = torch.cat(list_text_features)        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        # print("len", image_features.shape, self.zeroshot_weights.shape)
        # text_probs = (100.0 * image_features @ self.zeroshot_weights).softmax(dim=-1)
        # top_probs, top_labels = text_probs.topk(1, dim=-1)
        test_probs = image_features @ self.zeroshot_weights
        pred_cupl = torch.argmax(test_probs, dim =1)
        total_num_data += gt_labels.size(0)
        # print("ind", pred_cupl[0], gt_labels[0])
        total_correct += torch.sum(pred_cupl.unsqueeze(1) == gt_labels.unsqueeze(1)).item()
        avg_acc = total_correct / total_num_data
        ret = {"avg_acc": avg_acc}

        return ret