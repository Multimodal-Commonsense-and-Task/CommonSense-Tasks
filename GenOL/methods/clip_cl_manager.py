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
from utils.train_utils import select_model, select_cl_optimizer, select_scheduler
from utils.block_utils import MODEL_BLOCK_DICT, get_blockwise_flops

logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")

class CLIPCLManagerBase:
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

        assert kwargs["temp_batchsize"] <= kwargs["batchsize"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batch_size = kwargs["temp_batchsize"]
        self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.memory_size -= self.temp_batch_size
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
        print("here", self.total_samples)
        self.type_name = kwargs["type_name"]
        if self.model_name == 'clip_vit':
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes = get_transform(self.dataset, self.transforms, self.method_name, self.type_name, self.transform_on_gpu, 224)
        else:
            self.train_transform, self.test_transform, self.cpu_transform, self.n_classes = get_transform(self.dataset, self.transforms, self.method_name, self.type_name, self.transform_on_gpu)
        self.cutmix = "cutmix" in kwargs["transforms"]
        
        self.model, self.pretrain_train_transform, self.pretrain_val_transform, self.tokenizer, self.criterion = select_model(self.model_name, self.dataset)
        self.model = self.model.to(self.device)
        self.optimizer = select_cl_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        self.data_stream = iter(self.train_datalist)
        # self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)

        self.memory = MemoryBase(self.memory_size)
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
        self.n_model_cls = defaultdict(list)
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
        # self.get_flops_parameter()
        self.initialize_future()
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
        elif 'ImageNet' in self.dataset:
            self.tasks = 5
            self.cls_per_task = [200]*self.tasks
        elif 'CUB_200' in self.dataset:
            self.tasks = 5
            self.cls_per_task = [40]*self.tasks
        elif 'birds31' in self.dataset:
            self.tasks = 5
            self.cls_per_task = [5,5,5,5,6]
        
        total_cls=0
        self.cul_cls_per_task = []
        for i in range(self.tasks):
            total_cls+=self.cls_per_task[i]
            self.cul_cls_per_task.append(total_cls)
            
        self.cur_task = 0
        self.eval_point = [int(point) for point in eval_point.split(" ")]
        # self.prep_fast_adaptation_data()
        # self.fast_adaptation_samples_per_class = 100
        # self.fast_epoch = 10
        self.fast_trained = False
        self.aoa_eval = False
        self.saved_model = copy.deepcopy(self.model)
        self.fast_model = None
        
        self.no_eval_during_train = kwargs['no_eval_during_train']
        
        # self.prep_zero_shot_data()
        self.prompt_template = 'this is a photo of a '

    # def prep_zero_shot_data(self):
    #     self.exposed_classes = []
    #     self.cls_dict = {}
    #     self.cls_dict_opp = {}
    #     label = 0
    #     for data in self.train_datalist:
    #         if data["klass"] not in self.exposed_classes:
    #             self.exposed_classes.append(data["klass"])
    #             self.cls_dict[data["klass"]] = label
    #             self.cls_dict_opp[label] = data["klass"]
    #             label+=1
    
    
    def prep_fast_adaptation_data(self):
        self.p_cls_list = []
        self.p_cls_dict = {}
        label = 0
        for data in self.train_datalist:
            if data["klass"] not in self.p_cls_list:
                self.p_cls_list.append(data["klass"])
                self.p_cls_dict[data["klass"]] = label
                label+=1
            

    # Memory 새로 정의 (not MemoryBase)
    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker, self.test_transform)
        self.memory = MemoryBase(self.memory_size)

        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            
        if sample["time"] not in self.exposed_domains and "clear" in self.dataset:
            self.exposed_domains.append(sample["time"])
            
        self.temp_future_batch.append(sample)
        self.future_num_updates += self.online_iter

        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(int(self.future_num_updates))
            for stored_sample in self.temp_future_batch:
                self.update_memory(stored_sample)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def update_memory(self, sample):
        pass

    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch(aoa=self.aoa_eval)
        self.load_batch()
        return batch

    # stream 또는 memory를 활용해서 batch를 load해라
    # data loader에 batch를 전달해주는 함수
    def load_batch(self):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            self.dataloader.load_batch(self.waiting_batch[0])
            del self.waiting_batch[0]

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.memory_batch_size))

    def online_step(self, sample, sample_num, n_worker):
        self.fast_trained=False
        if self.fast_model is not None:
            self.fast_model = None
        
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []

    def add_new_class(self, class_name):
        # if hasattr(self.model, 'fc'):
        #     fc_name = 'fc'
        # elif hasattr(self.model, 'head'):
        #     fc_name = 'head'
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        
        self.text_class_prompts = [self.prompt_template+cla for cla in self.exposed_classes]
        self.text_class_tokens = self.tokenizer(self.text_class_prompts).to(self.device)
        
        # model_fc = getattr(self.model, fc_name)
        # prev_weight = copy.deepcopy(model_fc.weight.data)
        # prev_bias = copy.deepcopy(model_fc.bias.data)
        # setattr(self.model, fc_name, nn.Linear(model_fc.in_features, self.num_learned_class).to(self.device))
        # model_fc = getattr(self.model, fc_name)
        # with torch.no_grad():
        #     if self.num_learned_class > 1:
        #         model_fc.weight[:self.num_learned_class - 1] = prev_weight
        #         model_fc.bias[:self.num_learned_class - 1] = prev_bias
        # saved_model_fc = getattr(self.saved_model, fc_name)
        # saved_prev_weight = copy.deepcopy(saved_model_fc.weight.data)
        # saved_prev_bias = copy.deepcopy(saved_model_fc.bias.data)
        # setattr(self.saved_model, fc_name, nn.Linear(saved_model_fc.in_features, self.num_learned_class))
        # saved_model_fc = getattr(self.saved_model, fc_name)
        # with torch.no_grad():
        #     if self.num_learned_class > 1:
        #         saved_model_fc.weight[:self.num_learned_class - 1] = saved_prev_weight
        #         saved_model_fc.bias[:self.num_learned_class - 1] = saved_prev_bias
                
        # for param in self.optimizer.param_groups[1]['params']:
        #     if param in self.optimizer.state.keys():
        #         del self.optimizer.state[param]
        # del self.optimizer.param_groups[1]
        # self.optimizer.add_param_group({'params': model_fc.parameters()})
        # if 'reset' in self.sched_name:
        #     self.update_schedule(reset=True)

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            imgs = data["image"].to(self.device)
            labels = data["label"].to(self.device)
            # if self.aoa_eval and i%(self.online_iter*self.temp_batch_size)==0:
            #     aoa_x = data["not_aug_img"].to(self.device)
            #     self.aoa_evaluation(aoa_x, y)
                
            self.before_model_update()

            self.optimizer.zero_grad()

            image_features, text_features, logit_scale = self.model(imgs,self.text_class_tokens)

            loss = self.criterion(image_features, text_features, logit_scale, labels)
            # _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # self.total_flops += (len(y) * self.backward_flops)

            # self.after_model_update()s
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            itop_probs, top_labels = text_probs.topk(1, dim=-1)
            num_data += labels.size(0)

            total_loss += loss.item()
            correct += torch.sum(top_labels == labels.unsqueeze(1)).item()
            # num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def aoa_evaluation(self, image, label):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        self.saved_model.to(self.device)
        self.saved_model.eval()
        image, label = image[:self.temp_batch_size], label[:self.temp_batch_size]
        with torch.no_grad():
            logit = self.saved_model(image)
            pred = torch.argmax(logit, dim=-1)
            _, preds = logit.topk(self.topk, 1, True, True)

            total_correct += torch.sum(preds == label.unsqueeze(1)).item()
            total_num_data += label.size(0)
        avg_acc = total_correct / total_num_data
        logger.info(f"AOA | Sample # {self.sample_num} | Real Time Evaluation: {avg_acc:.3f}")
        
        self.saved_model = copy.deepcopy(self.model).cpu()

    def before_model_update(self):
        pass

    def after_model_update(self):
        self.update_schedule()

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y)

        # self.total_flops += (len(y) * self.forward_flops)
        return logit, loss

    def report_training(self, sample_num, train_loss, train_acc):
        # writer.add_scalar(f"train/loss", train_loss, sample_num)
        # writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, domain_name, sample_num, avg_acc, avg_loss, cls_acc):
        # writer.add_scalar(f"test/loss", avg_loss, sample_num)
        # writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"{domain_name} Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | cls_acc {cls_acc:.4f}"
        )

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def _save_ckpt(self, sample_num):
        save_path = f"pretrained/{self.dataset}/{self.type_name}/{self.online_iter}_{self.rnd_seed}"
        os.makedirs(f"pretrained/{self.dataset}/{self.type_name}/{self.online_iter}_{self.rnd_seed}", exist_ok=True)
        torch.save({
        'num_samples': sample_num,
        'tasks': self.exposed_classes,
        'model_state_dict': self.model.state_dict(),
        }, os.path.join(save_path, f"{sample_num}_ckpt.pth"))

    def online_evaluate(self, domain_name, cur_task, test_list, sample_num, batch_size, n_worker, cls_dict={}, cls_addition={}, time=0):
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
        
        self.report_test(domain_name, sample_num, eval_dict["avg_acc"], eval_dict["avg_loss"], eval_dict["cls_acc"])
            
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
        ret = {"avg_acc": avg_acc, "avg_loss": 0, "cls_acc": 0}

        return ret
    
    def fast_evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.fast_model(x)
                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()
                
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        avg_acc = total_correct / total_num_data
        if self.cur_task+1 == self.tasks:
            cls_acc_avg = cls_acc[self.cul_cls_per_task[self.cur_task]:]
        else:
            cls_acc_avg = cls_acc[self.cul_cls_per_task[self.cur_task]:self.cul_cls_per_task[self.cur_task+1]]
        return avg_acc, np.mean(cls_acc_avg)
        

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

    def _interpret_pred(self, y, pred):
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def calculate_fast_adaptation(self, domain_name, sample_num, test_list, cls_dict, batch_size, n_worker):
        if self.cur_task+1 == self.tasks:
            next_task_cls = self.p_cls_list[self.cul_cls_per_task[self.cur_task]:]
        else:
            next_task_cls = self.p_cls_list[self.cul_cls_per_task[self.cur_task]:self.cul_cls_per_task[self.cur_task+1]]
        
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(next_task_cls)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=next_task_cls,
            data_dir=self.data_dir,
            learned_classes=self.num_learned_class
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        
        if not self.fast_trained:
            train_data = []
            train_data_cnt = {}
            for data in self.train_datalist:
                if data["klass"] in next_task_cls:
                    if data["klass"] not in list(train_data_cnt.keys()):
                        train_data_cnt[data["klass"]] = 0
                    if train_data_cnt[data["klass"]] < self.fast_adaptation_samples_per_class:
                        train_data.append(data)
                        train_data_cnt[data["klass"]] += 1
            
            train_df = pd.DataFrame(train_data)
            train_dataset = ImageDataset(
                train_df,
                dataset=self.dataset,
                transform=self.test_transform,
                cls_list=next_task_cls,
                data_dir=self.data_dir,
                augmentation=self.train_transform,
                learned_classes=self.num_learned_class
            )
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
            )
            
            self.fast_trained=True
            self.fast_model = copy.deepcopy(self.model)
            
            if hasattr(self.fast_model, 'fc'):
                fc_name = 'fc'
            elif hasattr(self.fast_model, 'head'):
                fc_name = 'head'
            model_fc = getattr(self.fast_model, fc_name)
            prev_weight = copy.deepcopy(model_fc.weight.data)
            prev_bias = copy.deepcopy(model_fc.bias.data)
            setattr(self.fast_model, fc_name, nn.Linear(model_fc.in_features, self.cul_cls_per_task[self.cur_task+1]).to(self.device))
            model_fc = getattr(self.fast_model, fc_name)
            with torch.no_grad():
                model_fc.weight[:self.num_learned_class] = prev_weight
                model_fc.bias[:self.num_learned_class] = prev_bias
                    
            self.fast_optimizer = select_optimizer(self.opt_name, self.lr, self.fast_model)
            self.fast_scheduler = select_scheduler(self.sched_name, self.fast_optimizer)
            self.fast_model.train()
            for ep in range(self.fast_epoch):
                for i, data in enumerate(train_loader):
                    x = data["image"].to(self.device)
                    y = data["label"].to(self.device)

                    with torch.cuda.amp.autocast(self.use_amp):
                        logit = self.fast_model(x)
                        loss = self.criterion(logit, y)
                
                    _, preds = logit.topk(self.topk, 1, True, True)

                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.fast_optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.fast_optimizer.step()
            
        avg_acc, cls_acc = self.fast_evaluation(test_loader, self.criterion)
        logger.info(f"{domain_name} ADAPTATION | Sample # {sample_num} | Task{self.cur_task} -> Task{self.cur_task+1} fast adaptation: {avg_acc:.3f}")
        logger.info(f"{domain_name} IMBALANCE_ADAPT | Sample # {sample_num} | Task{self.cur_task} -> Task{self.cur_task+1} fast adaptation: {cls_acc:.3f}")
        del test_loader

    def calculate_task_metric(self, domain_name, sample_num, test_list, cls_dict, batch_size, n_worker):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        preds = []
        gts = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred.detach().cpu().numpy())
                gts.append(y.detach().cpu().numpy())
        self.preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        self.gt_label = gts
        self.calculate_task_acc(domain_name, sample_num)
        del test_loader
    
    def calculate_task_acc(self, domain_name, sample_num):
        correct_cnt = np.zeros(self.n_classes)
        cls_cnt = np.zeros(self.n_classes)
        for i, gt in enumerate(self.gt_label):
            cls_cnt[gt] += 1
            if gt == self.preds[i]:
                correct_cnt[gt] += 1
        correct_prob = correct_cnt/cls_cnt
        log = f'{domain_name} ACC_PER_TASK | Sample # {sample_num} | '
        task_acc=0
        task=0
        for cls in range(self.n_classes):
            task_acc+=correct_prob[cls]
            if cls+1 in self.cul_cls_per_task:
                log += f'Task {task}: {(task_acc/(self.cls_per_task[task])):.3f}, '
                task+=1
                task_acc=0
        
        logger.info(log)
    
    def get_forgetting(self, domain_name, sample_num, test_list, cls_dict, batch_size, n_worker):
        test_df = pd.DataFrame(test_list)
        test_dataset = ImageDataset(
            test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.p_cls_list,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        preds = []
        gts = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred.detach().cpu().numpy())
                gts.append(y.detach().cpu().numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        self.gt_label_forgetting = gts
        self.test_records[domain_name].append(preds)
        self.n_model_cls[domain_name].append(copy.deepcopy(self.num_learned_class))
        if len(self.test_records[domain_name]) > 1:
            klr, kgr = self.calculate_online_forgetting(self.n_classes, self.gt_label_forgetting, self.test_records[domain_name][-2], self.test_records[domain_name][-1], self.n_model_cls[domain_name][-2], self.n_model_cls[domain_name][-1])
            self.knowledge_loss_rate.append(klr)
            self.knowledge_gain_rate.append(kgr)
            self.forgetting_time.append(sample_num)
            logger.info(f'{domain_name} FORGETTING | KLR {klr} | KGR {kgr}')
            np.save(self.save_path + '_KLR.npy', self.knowledge_loss_rate)
            np.save(self.save_path + '_KGR.npy', self.knowledge_gain_rate)
            np.save(self.save_path + '_forgetting_time.npy', self.forgetting_time)


    def calculate_online_forgetting(self, n_classes, y_gt, y_t1, y_t2, n_cls_t1, n_cls_t2):
        total_cnt = len(y_gt)
        cnt_gt = np.zeros(n_classes)
        cnt_y1 = np.zeros(n_cls_t1)
        cnt_y2 = np.zeros(n_cls_t2)
        correct_y1 = np.zeros(n_classes)
        correct_y2 = np.zeros(n_classes)
        correct_both = np.zeros(n_classes)
        for i, gt in enumerate(y_gt):
            y1, y2 = y_t1[i], y_t2[i]
            cnt_gt[gt] += 1
            cnt_y1[y1] += 1
            cnt_y2[y2] += 1
            if y1 == gt:
                correct_y1[gt] += 1
                if y2 == gt:
                    correct_y2[gt] += 1
                    correct_both[gt] += 1
            elif y2 == gt:
                correct_y2[gt] += 1

        gt_prob = cnt_gt/total_cnt
        y1_prob = cnt_y1/total_cnt
        y2_prob = cnt_y2/total_cnt

        probs = np.zeros([n_classes, n_cls_t1, n_cls_t2])

        for i in range(n_classes):
            cls_prob = gt_prob[i]
            notlearned_prob = 1 - (correct_y1[i] + correct_y2[i] - correct_both[i])/cnt_gt[i]
            forgotten_prob = (correct_y1[i] - correct_both[i]) / cnt_gt[i]
            newlearned_prob = (correct_y2[i] - correct_both[i]) / cnt_gt[i]
            if i < n_cls_t1:
                marginal_y1 = y1_prob/(1-y1_prob[i])
                marginal_y1[i] = forgotten_prob/(notlearned_prob+1e-10)
            else:
                marginal_y1 = y1_prob
            if i < n_cls_t2:
                marginal_y2 = y2_prob/(1-y2_prob[i])
                marginal_y2[i] = newlearned_prob/(notlearned_prob+1e-10)
            else:
                marginal_y2 = y2_prob
            probs[i] = np.expand_dims(marginal_y1, 1) * np.expand_dims(marginal_y2, 0) * notlearned_prob * cls_prob
            if i < n_cls_t1 and i < n_cls_t2:
                probs[i][i][i] = correct_both[i]/total_cnt

        knowledge_loss = np.sum(probs*np.log(np.sum(probs, axis=(0, 1), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        knowledge_gain = np.sum(probs*np.log(np.sum(probs, axis=(0, 2), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=2, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        prob_gt_y1 = probs.sum(axis=2)
        prev_total_knowledge = np.sum(prob_gt_y1*np.log(prob_gt_y1/(np.sum(prob_gt_y1, axis=0, keepdims=True)+1e-10)/(np.sum(prob_gt_y1, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        max_knowledge = np.log(n_cls_t2)/np.log(n_classes)

        knowledge_loss_rate = knowledge_loss/prev_total_knowledge
        knowledge_gain_rate = knowledge_gain/(max_knowledge-prev_total_knowledge)
        return knowledge_loss_rate, knowledge_gain_rate
    
    def get_flops_parameter(self, method=None):
        _, _, _, inp_size, inp_channel = get_statistics(dataset=self.dataset, type_name=self.type_name)
        if self.model_name == 'vit':
            inp_size = 224
        
        flops_dict = get_model_complexity_info(self.model, (inp_channel, inp_size, inp_size),
                                                                             as_strings=False,
                                                                             print_per_layer_stat=False, verbose=True,
                                                                             criterion=self.criterion,
                                                                             original_opt=self.optimizer,
                                                                    opt_name=self.opt_name, lr=self.lr)
        forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops  = get_blockwise_flops(flops_dict, self.model_name, method)
        self.forward_flops = sum(forward_flops)
        self.backward_flops = sum(backward_flops)
        self.blockwise_forward_flops = forward_flops
        self.blockwise_backward_flops = backward_flops
        self.total_model_flops = self.forward_flops + self.backward_flops
        
        self.G_forward_flops, self.G_backward_flops = sum(G_forward_flops), sum(G_backward_flops)
        self.F_forward_flops, self.F_backward_flops = sum(F_forward_flops), sum(F_backward_flops)
        self.G_blockwise_forward_flops, self.G_blockwise_backward_flops = G_forward_flops, G_backward_flops
        self.F_blockwise_forward_flops, self.F_blockwise_backward_flops = F_forward_flops, F_backward_flops
        
         
    
class MemoryBase:
    
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.images = []
        self.labels = []
        self.update_buffer = ()
        self.cls_dict = dict()
        self.cls_list = []
        self.cls_count = []
        self.cls_idx = []
        self.usage_count = np.array([])
        self.class_usage_count = np.array([])
        self.current_images = []
        self.current_labels = []
        self.current_cls_count = [0 for _ in self.cls_list]
        self.current_cls_idx = [[] for _ in self.cls_list]

    def __len__(self):
        return len(self.images)

    def replace_sample(self, sample, idx=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
        else:
            assert idx < self.memory_size
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_dict[sample['klass']]
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.cls_list)
        self.cls_list.append(class_name)
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.class_usage_count = np.append(self.class_usage_count, 0.0)

    def retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
        return memory_batch
