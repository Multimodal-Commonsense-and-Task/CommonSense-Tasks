import os
import time
import datetime
import logging
import copy
import pandas as pd
import numpy as np
import torch
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
# from torch.utils.tensorboard import SummaryWriter

from utils.data_loader_VLM_base import MultiProcessLoader
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset, GenerationDataset
from utils.train_utils_VLM import get_VLMmodel
from utils.train_utils import CustomStoppingCriteria
from peft.tuners.lora import LoraLayer
import bitsandbytes

from utils.data_loader_VLM import GenerationDataset, DataCollatorForGenerationDataset
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num, can_infer
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from models.llava.mm_utils import KeywordsStoppingCriteria
from models.llava import conversation as conversation_lib_llava
from models.bunny import conversation as conversation_lib_bunny
from models.duallora.dualloralayer import DualLoraLayer
# writer = SummaryWriter("tensorboard")

from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
import numpy as np

from transformers.optimization import get_scheduler
from collections import OrderedDict
from utils.data_worker import ManagerWatchdog
import queue
from collections.abc import Mapping

from utils.eval_metrics import NLPEvaluator, matching_token_num

import json

class VLM: # Client
    def __init__(
        self,
        train_datalist, 
        test_datalist,
        device,
        data_args,
        model_args,
        args = None,
        bnb_model_from_pretrained_args=None,
    ):
        kwargs = vars(args)
        self.previous_param = defaultdict(int)
        self.args = args
        self.data_args = data_args
        self.model_args = model_args
        self.bnb_model_from_pretrained_args=bnb_model_from_pretrained_args
        self.device = device
        self.iter = 0
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.task_id = 0
        self.method_name = kwargs["mode"]
        self.memory_size = kwargs["memory_size"]
        self.online_iter = kwargs["online_iter"]
        self.zero_shot = kwargs["zero_shot"]
        print("zero_shot", self.zero_shot)
        self.lr = kwargs["learning_rate"]
        self.mm_projector_lr = kwargs["mm_projector_lr"]
        self.dataset = self.data_args.dataset
        print("self.dataset", self.dataset)
        assert kwargs["temp_batchsize"] <= kwargs["per_gpu_train_batch_size"]
        self.batch_size = kwargs["per_gpu_train_batch_size"]
        self.temp_batch_size = kwargs["temp_batchsize"]
        self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.memory_size -= self.temp_batch_size
        self.transforms = kwargs["transforms"]

        self.n_worker = kwargs["dataloader_num_workers"]
        self.future_steps = kwargs["future_steps"]

        self.eval_period = kwargs["eval_period"]
        self.topk = kwargs["topk"]
        self.f_period = kwargs["f_period"]

        self.seen_action_classes = []
        self.seen_object_classes = []
        self.seen_categories = []
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.cls_dict = {}
        self.total_samples = len(self.train_datalist)
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.sample_num = 0
        self.train_count = 0
        self.seen = 0
        
        logger = logging.getLogger()
        self.logger = logger
        
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.note = kwargs['note']
        self.rnd_seed = kwargs['seed']
        
        self.f_next_time = 0
        self.start_time = time.time()

        self.exposed_domains = []
        self.waiting_batch = []
        self.initialize_future()
        # self.init_model()
        self.total_flops = 0.0
        self.state = {}
        self.watchdog = ManagerWatchdog()
        
        self.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']
        self.gradient_checkpointing = kwargs['gradient_checkpointing']
        if self.gradient_checkpointing:
            gradient_checkpointing_kwargs = {'use_reentrant':False}
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                    
        # 576 for clip image encoder (llava)
        # 729 for siglip (bunny)
        if 'llava' in self.model_args.model_name_or_path.lower():
            self.img_feat_size = 576
        elif 'bunny' in self.model_args.model_name_or_path.lower():
            self.img_feat_size = 729
        
        
    def setup(self):
        model, tokenizer, data_args = get_VLMmodel(self.model_args, self.args, self.bnb_model_from_pretrained_args, self.data_args)
        self.model = model
        self.tokenizer = tokenizer
        self.data_args = data_args

        # max_steps = 8000 # FIXME
        self.create_optimizer()
        # self.create_scheduler(max_steps, optimizer=self.optimizer)

        # Activate gradient checkpointing if needed
        # if self.args.gradient_checkpointing: # default False
        #     if self.args.gradient_checkpointing_kwargs is None:
        #         gradient_checkpointing_kwargs = {}
        #     else:
        #         gradient_checkpointing_kwargs = self.args.gradient_checkpointing_kwargs

        #     self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    # from llava_traininer
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        print("self.args.mm_projector_lr", self.args.mm_projector_lr, "self.lr", self.lr)
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in self.model.named_parameters() if ("mm_projector" in name)]
                print("param1")
                print([
                        n for n, p in self.model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ])
                print()
                print("param2")
                print([
                        n for n, p in self.model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ])
                print()
                print("param3")
                print([
                        n for n, p in self.model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ])
                print()
                print("param4")
                print([
                        n for n, p in self.model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ])
                print()
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr":self.lr
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr":self.lr
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in self.model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        self.logger.info(f"bitsandbytes: will optimize {module} in fp32")
                print(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, num_cycles:int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        # if self.lr_scheduler is None:
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs={"num_cycles": num_cycles,}
        )
        self._created_lr_scheduler = True
        return self.lr_scheduler

    def get_lr(self):
        return self.lr

    @torch.no_grad()
    def init_model(self):
        # reinit vision tower & mm_projector of llava model
        if 'bunny' in self.model_args.model_name_or_path.lower() and '3b' in self.model_args.model_name_or_path.lower():
            self.model.load_state_dict(torch.load('./bunny3b_vision_tower_mm_projector.pth', map_location='cpu'), strict=False)
            self.logger.info("done loading init bunny 3b vision tower and mm projector\n")
        elif 'bunny' in self.model_args.model_name_or_path.lower() and '8b' in self.model_args.model_name_or_path.lower():
            self.model.load_state_dict(torch.load('./bunny8b_vision_tower_mm_projector.pth', map_location='cpu'), strict=False)
            self.logger.info("done loading init bunny 8b vision tower and mm projector\n")
        elif 'llava' in self.model_args.model_name_or_path.lower():
            self.model.load_state_dict(torch.load('./llava_vision_tower_mm_projector.pth', map_location='cpu'), strict=False)
            self.logger.info("done loading init llava vision tower and mm projector\n")
        else:
            raise ValueError("wrong model name")
        # reset lora layers
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters('default', True)
        print("self.model")
        print(self.model)
        self.logger.info("done reset lora layers\n")

    # Memory 새로 정의 (not MemoryBase)
    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.setup()
        self.dataloader = MultiProcessLoader(self.n_worker, self.device, tokenizer=self.tokenizer, data_args=self.data_args)
        self.memory = MemoryBase(self.memory_size)

        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True
        
        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
            
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
        self.reservoir_memory(sample)

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch()
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

    def online_step(self, sample, sample_num):
        self.sample_num = sample_num
        if self.dataset == "Bongard-HOI":
            if sample["object_class"] not in self.seen_object_classes and sample["action_class"] not in self.seen_action_classes:
                self.seen_object_classes.append(sample["object_class"])
                self.seen_action_classes.append(sample["action_class"])
        elif self.dataset == "Bongard-Openworld":
            if sample["commonSense"] not in self.seen_categories:
                self.seen_categories.append(sample["commonSense"])
            
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                if not self.zero_shot:
                    train_loss = self.online_train(iterations=int(self.num_updates))
                    self.report_training(sample_num, train_loss)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []

    def _prepare_input(self, data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                "training dataset contains keys expected by the model"
            )
        return inputs
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


    # def online_train(self, iterations=1):
    #     total_loss, correct, num_data = 0.0, 0.0, 0.0
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     for i in range(iterations):
    #         self.iter += 1
    #         data = self.get_batch()
    #         data = self._prepare_inputs(data)
    #         loss = self.compute_loss(self.model, data)
    #         loss.backward()
    #         for name, param in self.model.named_parameters():
    #             if param.requires_grad:
    #                 self.previous_param[name] = copy.deepcopy(param)
            
    #         # gradient check option      
    #         # for param in self.model.parameters():
    #         #     if param.requires_grad:
    #         #         print(param.grad.view(-1))
                         
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()
            
    #         # for name, param in self.model.named_parameters():
    #         #     if param.requires_grad:
    #         #         check_tensor = self.previous_param[name] != param
    #         #         if torch.sum(check_tensor.long()) > 0:
    #         #             print(name, "check", torch.sum(check_tensor.long()))
    #         #             self.previous_param[name] = param   

    #         total_loss += loss.item()
    #     return total_loss / iterations


    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        self.optimizer.zero_grad()
        for i in range(iterations):
            self.iter += 1
            data = self.get_batch()
            data = self._prepare_inputs(data)
            loss = self.compute_loss(self.model, data)
            loss /= self.gradient_accumulation_steps
            loss.backward()
            
            if (self.iter) % self.gradient_accumulation_steps == 0:
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         self.previous_param[name] = param
                self.optimizer.step()
                self.optimizer.zero_grad()
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         #if type(self.previous_param[name]) != int:
                #         check_tensor = self.previous_param[name] != param
                #         if torch.sum(check_tensor.long()) > 0:
                #             print(name, "check", torch.sum(check_tensor.long()))
                #         self.previous_param[name] = param   
            total_loss += loss.item()
        return total_loss / iterations

    def report_training(self, sample_num, train_loss):
        self.logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | LR {self.get_lr()} |"
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, scores, dataname):
        self.logger.info(
            f"Test | Sample # {sample_num} | Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |"
        )      

    def online_evaluate(self, test_datalist, iteration):
        test_df = pd.DataFrame(test_datalist)
        if self.dataset == "Bongard-HOI":
            seen_test_df = test_df[(test_df['action_class'].isin(self.seen_action_classes)) & (test_df['object_class'].isin(self.seen_object_classes))]
        elif self.dataset == "Bongard-Openworld":
            seen_test_df = test_df[test_df['commonSense'].isin(self.seen_categories)]
        print("eval set size", len(seen_test_df))

        seen_testdata_list = []
        for item_idx, row in enumerate(seen_test_df.values):
            dic = {}
            for key_idx, key in enumerate(seen_test_df.keys()):
                dic[key] = row[key_idx]
            seen_testdata_list.append(dic)

        dataset = GenerationDataset(seen_testdata_list, self.tokenizer, self.data_args)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=2, drop_last=False, collate_fn=DataCollatorForGenerationDataset(self.tokenizer))
        
        if 'llava' in self.model_args.model_name_or_path.lower():
            conv = conversation_lib_llava.default_conversation
        elif 'bunny' in self.model_args.model_name_or_path.lower():
            conv = conversation_lib_bunny.default_conversation
        repeat_criteria = CustomStoppingCriteria()
        stop_str = conv.sep2
        keywords = [stop_str]
        
        # img_feat_size = 729
        self.model.eval()
        predictions = []
        n_word_total = 0
        n_generated_word_total = 1
        n_word_correct = 1
        cnt = 0
        with torch.no_grad():
            # for i, (inputs, imgs, golds, prompts, img_files) in enumerate(tqdm(dataloader)):
            for i, batch in enumerate(tqdm(dataloader)):
                inputs, imgs, golds, prompts, img_files = batch['input_ids'], batch['images'], batch['gold'], batch['prompt'], batch['image_file']
                attention_mask = batch['attention_mask'].to(device=self.device)
                
                inputs = inputs.to(device=self.device, non_blocking=True)
                if imgs is not None:
                    if isinstance(imgs, list):
                        imgs = [img.to(device=self.device, dtype=torch.bfloat16, non_blocking=True) for img in imgs]
                    else:
                        imgs = imgs.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    image_sizes = [x.shape[-2:] for x in imgs]
                keyword_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, inputs)
                stopping_criteria = StoppingCriteriaList([repeat_criteria, keyword_criteria])
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        images=imgs,
                        do_sample=True,# if args.temperature > 0 else False,
                        temperature=self.args.eval_temp,#args.temperature,
                        top_p=None,#args.top_p,
                        num_beams=1,#args.num_beams,
                        max_new_tokens=self.model_args.max_new_tokens,#args.max_new_tokens,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        stopping_criteria = stopping_criteria
                    )
                
                pred_sentences = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0].strip()
                for pred_sentence, gold, prompt, img_file in zip(pred_sentences, golds, prompts, img_files):
                    pred_sentence = pred_sentence.strip()
                    input_label = self.tokenizer.encode(gold)
                    output_id = self.tokenizer.encode(pred_sentence)
                    n_word = len(set(input_label))
                    n_generated_word = len(set(output_id))
                    n_correct = matching_token_num(output_id, input_label)
                    predictions.append({"image_file":img_file, "input":prompt, "sentence":pred_sentence, "gt_sentence":gold.strip()})
                                        
                    n_word_total += n_word
                    n_generated_word_total += n_generated_word
                    n_word_correct += n_correct
                    cnt += 1
                    
        scores = NLPEvaluator(predictions).evaluate()
        scores["precision"] = n_word_correct / n_word_total
        scores["recall"] = n_word_correct / n_generated_word_total
        
        predictions.append(scores)
        #save predictions
        self.logger.info(f"Test precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
        os.makedirs(f"./eval_results/{self.args.mode}", exist_ok=True)
        with open(f"./eval_results/{self.args.mode}/{self.args.note}_iter{iteration}.json", 'w') as fp:
            json.dump(predictions, fp, indent=4)

    
class MemoryBase:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.images = []
        self.labels = []

        self.update_buffer = ()
        
        self.usage_count = np.array([])
        self.current_images = []
        self.current_labels = []

    def __len__(self):
        return len(self.images)

    def replace_sample(self, sample, idx=None):
        if idx is None:
            assert len(self.images) < self.memory_size
            self.images.append(sample)
        else:
            assert idx < self.memory_size
            self.images[idx] = sample

    def retrieval(self, size, return_index=False):
        sample_size = min(size, len(self.images))
        memory_batch = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
        if return_index:
            return memory_batch, indices
        else:
            return memory_batch