import logging
import os
import copy
import random

import time
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flops_counter.ptflops import get_model_complexity_info
from utils.data_loader import ImageTextDataset, cutmix_data, MultiProcessLoader, get_statistics
from utils.augment import get_transform
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.block_utils import MODEL_BLOCK_DICT, get_blockwise_flops

logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")

class ZeroShotClip:
    def __init__(self, train_datalist, test_datalist, device, **kwargs):

        self.device = device

        self.method_name = kwargs["mode"]
        self.dataset = kwargs["dataset"]
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.init_cls = kwargs["init_cls"]
        
        self.memory_size = kwargs["memory_size"]
        self.online_iter = kwargs["online_iter"]

        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'const'
        self.lr = kwargs["lr"]
        # self.block_names = MODEL_BLOCK_DICT[self.model_name]
        # self.num_blocks = len(self.block_names) - 1

        # assert kwargs["temp_batchsize"] <= kwargs["batchsize"]
        self.batch_size = kwargs["batchsize"]
        # self.temp_batch_size = kwargs["temp_batchsize"]
        # self.memory_batch_size = self.batch_size - self.temp_batch_size
        # self.memory_size -= self.temp_batch_size
        self.transforms = kwargs["transforms"]

        self.criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        self.data_dir = kwargs["data_dir"]
        if self.data_dir is None:
            self.data_dir = os.path.join("dataset", self.dataset)
        self.n_worker = kwargs["n_worker"]
        self.future_steps = kwargs["future_steps"]
        self.transform_on_gpu = kwargs["transform_on_gpu"]
        self.use_kornia = kwargs["use_kornia"]
        self.transform_on_worker = kwargs["transform_on_worker"]

        self.eval_period = kwargs["eval_period"]
        self.topk = kwargs["topk"]
        self.f_period = kwargs["f_period"]

        self.use_amp = kwargs["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.cls_dict = {}
        self.total_samples = len(self.train_datalist)
        self.type_name = kwargs["type_name"]
        if self.model_name == 'clip_vit':
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes = get_transform(self.dataset, self.transforms, self.method_name, self.type_name, self.transform_on_gpu, 224)
        else:
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes = get_transform(self.dataset, self.transforms, self.method_name, self.type_name, self.transform_on_gpu)
        self.cutmix = "cutmix" in kwargs["transforms"]
        
        self.model, self.pretrain_train_transform, self.pretrain_val_transform, self.tokenizer, self.criterion = select_model(self.model_name, self.dataset)
        self.model = self.model.to(self.device)
        # self.beta1, self.beta2, self.eps, self.wd = kwargs["beta1"], kwargs["beta2"], kwargs["eps"], kwargs["weight_decay"]
        # self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, self.beta1, self.beta2, self.eps, self.wd )
        # self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr)
        # self.zeroshot_evaldataset = kwargs["zeroshot_evaldataset"]

        self.data_stream = iter(self.train_datalist)
        # self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)

        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.sample_num = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0

        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.gt_label = None
        self.gt_label_forgetting = None
        self.test_records = defaultdict(list)
        self.n_model_cls = []
        self.knowledge_loss_rate = []
        self.knowledge_gain_rate = []
        self.forgetting_time = []
        self.note = kwargs['note']
        self.rnd_seed = kwargs['rnd_seed']
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        self.f_period = kwargs['f_period']
        self.f_next_time = 0
        self.start_time = time.time()
        #num_samples = {'cifar10': 10000, 'PACS':1670, 'cifar10_10': 10000, 'PACS_10':16700, "OfficeHome":4357, "DomainNet":50872}
        self.total_samples = len(train_datalist)#num_samples[self.dataset]

        self.exposed_domains = []
        self.waiting_batch = []
        self.total_flops = 0.0
        # self.writer = SummaryWriter(f'tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}')
        
        eval_point = kwargs['eval_point']
        if 'clear' in self.dataset:
            self.tasks = 10
        elif 'cct' in self.dataset:
            self.tasks = 4
            self.cls_per_task = [3]*self.tasks
        elif 'PACS' in self.dataset:
            self.tasks = 3
            self.cls_per_task = [3,2,2]
        elif 'DomainNet' in self.dataset:
            self.tasks = 5
            self.cls_per_task = [69]*self.tasks
        elif 'cifar10' in self.dataset:
            self.tasks = 5
            self.cls_per_task = [2]*self.tasks
        elif 'NICO' in self.dataset:
            self.tasks = 5
            self.cls_per_task = [12]*self.tasks
        
        total_cls=0
        for i in range(self.tasks):
            total_cls+=self.cls_per_task[i]
            self.cls_per_task[i] = total_cls
            
        self.cur_task = 0
        self.eval_point = [int(point) for point in eval_point.split(" ")]
        self.prep_zero_shot_data()
        self.fast_adaptation_samples_per_class = 100
        self.fast_epoch = 10
        self.fast_trained = False
        self.aoa_eval = True
        self.saved_model = copy.deepcopy(self.model)
        self.fast_model = None
        
        self.prompt_template = 'this is a photo of a '
        self.text_class_prompts = [self.prompt_template+cla for cla in self.exposed_classes]
        self.text_class_tokens = self.tokenizer(self.text_class_prompts).to(self.device)
        

    def prep_zero_shot_data(self):
        self.exposed_classes = []
        self.cls_dict = {}
        self.cls_dict_opp = {}
        label = 0
        for data in self.train_datalist:
            if data["klass"] not in self.exposed_classes:
                self.exposed_classes.append(data["klass"])
                self.cls_dict[data["klass"]] = label
                self.cls_dict_opp[label] = data["klass"]
                label+=1
        
    def report_test(self, domain_name, sample_num, avg_acc):
        print(avg_acc, sample_num)
        # writer.add_scalar(f"test/acc", "avg_TR1", avg_TR1, sample_num)
        logger.info(
            f"{domain_name} Test | Sample # {sample_num} | test_acc {avg_acc:.4f}"
        )

    def online_evaluate(self, domain_name, cur_task, test_list, sample_num, batch_size, n_worker, train_list=None):
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

        self.model.eval()
        list_img_features = []
        list_text_features = []
        gt_labels = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                imgs = data["image"]
                labels = data["label"]
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                image_features = self.model.encode_image(imgs)
                text_features = self.model.encode_text(self.text_class_tokens)
                
                list_img_features.append(image_features.cpu())
                list_text_features.append(text_features.cpu())
                gt_labels.append(labels.cpu())
        
        gt_labels = torch.cat(gt_labels)
        image_features = torch.cat(list_img_features)
        text_features = torch.cat(list_text_features)        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
                    
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.topk(1, dim=-1)
        total_num_data += gt_labels.size(0)
        
        total_correct += torch.sum(top_labels == gt_labels.unsqueeze(1)).item()

        avg_acc = total_correct / total_num_data
        ret = {"avg_acc": avg_acc}

        return ret


    