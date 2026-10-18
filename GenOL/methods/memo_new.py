import logging
import copy
import types
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.train_utils import select_model, select_optimizer, select_scheduler
from collections import defaultdict
from utils.data_loader import ImageDataset
from utils.data_loader import MultiProcessLoader
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

from torch import optim


class MEMO(CLManagerBase):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)   
        self.save_prev_class_num = False
        self.prev_num_learned_class = None
        self.samples_per_task = kwargs["samples_per_task"]
        
        self.mean = defaultdict(int)
        self.iteration = 0
        
        
    def initialize_future(self):
        self.model = self.model.cpu()
        if "resnet" in self.model_name:
            self.model = MEMOResnet(self.model_name, self.dataset).to(self.device)
        else:
            self.model = MEMOVIT(self.model_name, self.dataset).to(self.device)
        self.model.num_features = self.model.fc.in_features
        if "resnet" in self.model_name:
            self.extractor = select_model(self.model_name, self.dataset, 1, F=True, ver2=True)
            self.extractor.fc = nn.Identity()
        else:
            _, self.extractor = select_model(self.model_name, self.dataset, 1, F=True, ver2=True)
            self.extractor[-1] = nn.Identity()
        self.model.AdaptiveExtractors.append(copy.deepcopy(self.extractor.to(self.device)))
        for n, p in self.model.named_parameters():
            print(n)
        # self.get_flops_parameter(self.method_name)

        
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.num_features = self.model.fc.in_features
        
        self.memory = MEMOMemory(self.memory_size, self.model_name, self.dataset)
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker, test_transform=self.test_transform)
        
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.temp_future_batch_idx = []
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
    
    def balanced_replace_memory(self, sample):
        if len(self.memory.images) >= self.memory.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.memory.cls_dict[sample['klass']]] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)
    
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
                self.balanced_replace_memory(stored_sample)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0
        
    
    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        
        update_aux_param = False
        if len(self.model.AdaptiveExtractors)>1: 
            if self.model.aux_classifier:
                aux_prev_weight = copy.deepcopy(self.model.aux_classifier.weight.data)
                aux_prev_bias = copy.deepcopy(self.model.aux_classifier.bias.data)
                update_aux_param = True
            prev_weight = copy.deepcopy(self.model.fc.weight.data)
            prev_bias = copy.deepcopy(self.model.fc.bias.data)
            
            self.model.aux_classifier = nn.Linear(self.num_features, self.num_learned_class-self.prev_num_learned_class+1).to(self.device)
            self.model.fc = nn.Linear(self.out_dim, self.num_learned_class).to(self.device)
            
            with torch.no_grad():
                # print("shape", prev_weight.shape)
                self.model.fc.weight[:prev_weight.shape[0], :prev_weight.shape[1]] = prev_weight
                self.model.fc.bias[:prev_weight.shape[0]] = prev_bias
                
                if update_aux_param:
                    self.model.aux_classifier.weight[:aux_prev_weight.shape[0], :aux_prev_weight.shape[1]] = aux_prev_weight
                    self.model.aux_classifier.bias[:aux_prev_weight.shape[0]] = aux_prev_bias
            
            # for real time evaluation
            saved_prev_weight = copy.deepcopy(self.saved_model.fc.weight.data)
            saved_prev_bias = copy.deepcopy(self.saved_model.fc.bias.data)
            self.saved_model.fc = nn.Linear(self.out_dim, self.num_learned_class)
            with torch.no_grad():
                # print("shape", prev_weight.shape)
                self.saved_model.fc.weight[:saved_prev_weight.shape[0], :saved_prev_weight.shape[1]] = saved_prev_weight
                self.saved_model.fc.bias[:saved_prev_weight.shape[0]] = saved_prev_bias
            
            if update_aux_param:      
                for param in self.optimizer.param_groups[2]['params']:
                    if param in self.optimizer.state.keys():
                        del self.optimizer.state[param]
                del self.optimizer.param_groups[2]

            for param in self.optimizer.param_groups[1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[1]
                
            self.optimizer.add_param_group({'params': self.model.fc.parameters()})
            self.optimizer.add_param_group({'params': self.model.aux_classifier.parameters()})
            if 'reset' in self.sched_name:
                self.update_schedule(reset=True)
        
            
        else:
            prev_weight = copy.deepcopy(self.model.fc.weight.data)
            prev_bias = copy.deepcopy(self.model.fc.bias.data)
            self.model.fc = nn.Linear(self.num_features, self.num_learned_class).to(self.device)
            with torch.no_grad():
                if self.num_learned_class > 1:
                    self.model.fc.weight[:prev_weight.shape[0], :self.num_features] = prev_weight
                    self.model.fc.bias[:prev_weight.shape[0]] = prev_bias
        
            for param in self.optimizer.param_groups[1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[1]
            self.optimizer.add_param_group({'params': self.model.fc.parameters()})
            if 'reset' in self.sched_name:
                self.update_schedule(reset=True)
        
    def aoa_evaluation(self, image, label):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        self.saved_model.to(self.device)
        self.saved_model.eval()
        image, label = image[:self.temp_batch_size], label[:self.temp_batch_size]
        with torch.no_grad():
            logit, _ = self.saved_model(image, test=True)
            pred = torch.argmax(logit, dim=-1)
            _, preds = logit.topk(self.topk, 1, True, True)

            total_correct += torch.sum(preds == label.unsqueeze(1)).item()
            total_num_data += label.size(0)
        avg_acc = total_correct / total_num_data
        logger.info(f"AOA | Sample # {self.sample_num} | Real Time Evaluation: {avg_acc:.3f}")
        
        self.saved_model = copy.deepcopy(self.model).cpu()    
    
    def online_train(self, iterations=1):
    
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        for i in range(iterations):
            self.iteration += 1
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            
            if self.aoa_eval and i%(self.online_iter*self.temp_batch_size)==0:
                aoa_x = data["not_aug_img"].to(self.device)
                self.aoa_evaluation(aoa_x, y)
                
            
            self.before_model_update()

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(self.use_amp):
                cls_pred, aux_cls_pred = self.model(x)
                
                # self.total_flops += len(x) * (self.forward_flops + self.backward_flops)
                # self.total_flops += len(x) * (self.F_forward_flops + self.F_backward_flops) * (len(self.model.AdaptiveExtractors) - 1)

                loss_clf = self.criterion(cls_pred, y)
                loss = loss_clf
                if aux_cls_pred is not None:
                    aux_y = y.clone()
                    aux_y = torch.where(aux_y-self.prev_num_learned_class>0, aux_y-self.prev_num_learned_class+1, 0)
                    loss_aux = self.criterion(aux_cls_pred, aux_y)
                    loss += loss_aux

            _, preds = cls_pred.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.after_model_update()

            total_loss += loss.item() 
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            
        return total_loss / iterations, correct / num_data
            
    
            
    def online_after_task(self):
        
        print("update prev_num_learned_class")
        self.prev_num_learned_class = self.num_learned_class
        self.model.aux_classifier = None
        print("extractor added")
        self.model.AdaptiveExtractors.append(copy.deepcopy(self.extractor.to(self.device)))
        self.model.AdaptiveExtractors[-1].load_state_dict(self.model.AdaptiveExtractors[-2].state_dict())

        for _, param in self.model.AdaptiveExtractors[-2].named_parameters():
            param.requires_grad = False
        # self.memory.update_after_task()
        
        self.out_dim = len(self.model.AdaptiveExtractors)*self.num_features
        self.model.norm = nn.LayerNorm(self.out_dim).to(self.device)

        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.out_dim, self.num_learned_class).to(self.device)
        with torch.no_grad():
            self.model.fc.weight[:prev_weight.shape[0], :prev_weight.shape[1]] = prev_weight
            self.model.fc.bias[:prev_weight.shape[0]] = prev_bias
        
        self.saved_model = copy.deepcopy(self.model).cpu()
        
        if len(self.optimizer.param_groups) == 2:
            for param in self.optimizer.param_groups[1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[1]
            self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        else:
            for param in self.optimizer.param_groups[2]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[2]
            for param in self.optimizer.param_groups[1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[1]
            self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
    
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
                logit, _ = self.model(x, test=True)
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
    
    
    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        feature_dict = {}
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                cls_pred, _ = self.model(x, test=True)

                loss = criterion(cls_pred, y)
                pred = torch.argmax(cls_pred, dim=-1)
                _, preds = cls_pred.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
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
                logit, _ = self.fast_model(x, test=True)

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

        avg_acc = total_correct / total_num_data
        return avg_acc
    
    def calculate_fast_adaptation(self, domain_name, sample_num, test_list, cls_dict, batch_size, n_worker):
        if self.cur_task+1 == self.tasks:
            next_task_cls = self.p_cls_list[self.cls_per_task[self.cur_task]:]
        else:
            next_task_cls = self.p_cls_list[self.cls_per_task[self.cur_task]:self.cls_per_task[self.cur_task+1]]
        
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
            setattr(self.fast_model, fc_name, nn.Linear(model_fc.in_features, self.cls_per_task[self.cur_task+1]).to(self.device))
            model_fc = getattr(self.fast_model, fc_name)
            with torch.no_grad():
                model_fc.weight[:self.num_learned_class] = prev_weight
                model_fc.bias[:self.num_learned_class] = prev_bias
                    
            self.fast_optimizer = select_optimizer(self.opt_name, self.lr, self.fast_model)
            self.fast_scheduler = select_scheduler(self.sched_name, self.fast_optimizer)
            
            for ep in range(self.fast_epoch):
                for i, data in enumerate(train_loader):
                    x = data["image"].to(self.device)
                    y = data["label"].to(self.device)
                    with torch.cuda.amp.autocast(self.use_amp):
                        logit, _ = self.fast_model(x)
                        loss = self.criterion(logit, y)
                
                    _, preds = logit.topk(self.topk, 1, True, True)

                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.fast_optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.fast_optimizer.step()
            
        avg_acc = self.fast_evaluation(test_loader, self.criterion)
        logger.info(f"{domain_name} ADAPTATION | Sample # {sample_num} | Task{self.cur_task} -> Task{self.cur_task+1} fast adaptation: {avg_acc:.3f}")
    
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
                logit, _ = self.model(x, test=True)
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred.detach().cpu().numpy())
                gts.append(y.detach().cpu().numpy())
        self.preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        self.gt_label = gts
        self.calculate_task_acc(domain_name, sample_num)


class MEMOResnet(nn.Module):
    def __init__(self, model_name, dataset):
        super(MEMOResnet, self).__init__()
        self.backbone = select_model(model_name, dataset, 1, G=True, ver2=True)
        self.extractor = select_model(model_name, dataset, 1, F=True, ver2=True)
        self.fc = nn.Linear(self.extractor.fc.in_features, 1)
        self.extractor.fc = nn.Identity()
        self.AdaptiveExtractors = nn.ModuleList()
        self.aux_classifier = None
        self.num_features = None
    
    def forward(self, x, test=False):
        aux_cls_out = None
        out0 = self.backbone(x)
        features = [extractor(out0) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features,1)
        cls_out = self.fc(features)
        if self.aux_classifier is not None and not test:
            aux_cls_out = self.aux_classifier(features[:,-self.num_features:])
        return cls_out, aux_cls_out
    
class MEMOVIT(nn.Module):
    def __init__(self, model_name, dataset):
        super(MEMOVIT, self).__init__()
        self.backbone = select_model(model_name, dataset, 1, G=True, ver2=True)
        self.norm, self.extractor = select_model(model_name, dataset, 1, F=True, ver2=True)
        self.fc = nn.Linear(self.extractor[-1].in_features, 1)
        self.aux_norm = nn.LayerNorm(self.extractor[-1].in_features)
        self.extractor[-1] = nn.Identity()
        self.AdaptiveExtractors = nn.ModuleList()
        self.aux_classifier = None
        self.num_features = None
    
    def forward(self, x, test=False):
        aux_cls_out = None
        out0 = self.backbone(x)
        print("len(AdaptiveExtractors)", len(self.AdaptiveExtractors))
        print("self.fc.in_features", self.fc.in_features)
        features = [extractor(out0) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features,2)
        features = features.mean(dim=1)
        features = self.norm(features)
        cls_out = self.fc(features)
        if self.aux_classifier is not None and not test:
            features = features[:,-self.num_features:]
            aux_cls_out = self.aux_classifier(self.aux_norm(features))
        return cls_out, aux_cls_out

class MEMOMemory(MemoryBase):
    def __init__(self, baseinit_mem_size, model_name, dataset):
        self.model_name = model_name
        self.sample_id = 0
        self.sample_ids = []
        self.dataset = dataset
        super().__init__(baseinit_mem_size)
    
    def replace_sample(self, sample, idx=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(self.sample_id)
            self.images.append(sample)
            self.sample_ids.append(self.sample_id)
        else:
            assert idx < self.sample_id
            remove_ind = self.sample_ids.index(idx)
            remove_label = self.cls_dict[self.images[remove_ind]['klass']]
            
            del self.images[remove_ind]
            del self.sample_ids[remove_ind]
            self.cls_count[remove_label] -= 1
            self.cls_idx[remove_label].remove(idx)
            
            self.images.append(sample)
            self.sample_ids.append(self.sample_id)
            self.cls_idx[self.cls_dict[sample['klass']]].append(self.sample_id)
        self.sample_id += 1
    
    def update_after_task(self):
        
        if "18" in self.model_name:
            if "clear" in self.dataset or "imagenet" in self.dataset:
                self.memory_size -= 223
            else:
                raise NotImplementedError("No memory size reduction")
        elif "32" in self.model_name:
            if "cifar" in self.dataset:
                self.memory_size -= 457
            elif "tinyimagenet" == self.dataset:
                self.memory_size -= 114
            elif "clear" in self.dataset or "imagenet" in self.dataset:
                self.memory_size -= 9
            else:
                raise NotImplementedError("No memory size reduction")
        if sum(self.cls_count)>self.memory_size:
            self.primary_reduction()
            self.secondary_reduction()
        
    def primary_reduction(self):
        min_images = torch.tensor((self.memory_size//len(self.cls_list))+1)
        cls_to_be_reduced = torch.where(min_images<torch.tensor(self.cls_count))[0]

        for cls in cls_to_be_reduced:
            removed_samples_id = np.random.choice(self.cls_idx[cls.item()], size=(self.cls_count[cls.item()]-min_images).item(), replace=False)
            removed_samples_ind = torch.where(torch.isin(torch.tensor(self.sample_ids), torch.tensor(removed_samples_id)))[0]
            mask = torch.ones(len(self.sample_ids), dtype=torch.bool)
            mask[torch.tensor(removed_samples_ind)] = 0
            self.sample_ids = torch.index_select(torch.tensor(self.sample_ids), 0, torch.nonzero(mask)[:,0])
            self.images = [item for i, item in enumerate(self.images) if i not in removed_samples_ind]
            self.cls_idx[cls.item()] = list(set(self.cls_idx[cls.item()])-set(removed_samples_id))
            self.cls_count[cls.item()] = min_images.item()
            self.sample_ids = self.sample_ids.tolist()
    
    def secondary_reduction(self):
        while sum(self.cls_count)>self.memory_size:
            remove_cls = self.cls_count.index(max(self.cls_count))
            cls_id = np.random.choice(self.cls_idx[remove_cls], size=1, replace=False)
            remove_ind = self.sample_ids.index(cls_id[0])
            self.cls_idx[remove_cls].remove(cls_id[0])
            self.cls_count[remove_cls] -= 1
            del self.images[remove_ind]
            del self.sample_ids[remove_ind]
