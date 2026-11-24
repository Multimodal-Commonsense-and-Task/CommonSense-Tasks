import logging.config
import os
import random
from scipy.special import softmax

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from typing import Dict, List
import transformers.trainer_utils as tu
from trl.rewards import think_format_reward


def seed_worker(worker_id, *args, **kwargs):
    worker_seed = torch.initial_seed() % 2**32
    transformers.set_seed(worker_seed)

tu.seed_worker = seed_worker

from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_task_vectors, load_deepspeed

from methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset, Qwen_LazySupervisedDataset, Qwen_DataCollatorForSupervisedDataset, Intern_LazySupervisedDataset, Intern_DataCollatorForSupervisedDataset, Onevision_LazySupervisedDataset, Onevision_DataCollatorForSupervisedDataset, Idefics_LazySupervisedDataset, Idefics_DataCollatorForSupervisedDataset, Otter_LazySupervisedDataset, Otter_DataCollatorForSupervisedDataset, OneVision1_5_LazySupervisedDataset, OneVision1_5_DataCollatorForSupervisedDataset, QwenVL3_LazySupervisedDataset, QwenVL3_DataCollatorForSupervisedDataset, Intern3_5_LazySupervisedDataset, Intern3_5_DataCollatorForSupervisedDataset
from utils.data_loader_VLM import make_inputs_dataloader
import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

from models.coda_prompt import CodaPrompt
from collections import OrderedDict

# from utils.selection_methods import sample_selection
import json
import bisect
from collections import defaultdict
import gc


class WallClockStepTimingCallback(transformers.TrainerCallback):
    """Record synchronized per-step wall time with sub-second precision."""

    def __init__(self, output_path, stop_after_steps=None):
        self.output_path = os.path.abspath(output_path)
        self.stop_after_steps = stop_after_steps
        self._handle = None
        self._started_ns = None
        self._previous_step_end_ns = None

    @staticmethod
    def _synchronize_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_begin(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._handle = open(self.output_path, "w", buffering=1)
        self._synchronize_cuda()
        self._previous_step_end_ns = time.perf_counter_ns()

    def on_step_begin(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        self._synchronize_cuda()
        self._started_ns = time.perf_counter_ns()

    def on_step_end(self, args, state, control, **kwargs):
        # [p40] every rank must stop at the same step, or the others hang in the next collective
        if (
            args.process_index != 0
            and self.stop_after_steps is not None
            and state.global_step >= self.stop_after_steps
        ):
            control.should_training_stop = True
            control.should_epoch_stop = True
            return control
        if args.process_index != 0 or self._started_ns is None:
            return
        self._synchronize_cuda()
        ended_ns = time.perf_counter_ns()
        active_seconds = (ended_ns - self._started_ns) / 1e9
        cycle_seconds = (ended_ns - self._previous_step_end_ns) / 1e9
        self._handle.write(
            json.dumps(
                {
                    "step": int(state.global_step),
                    "cycle_seconds": cycle_seconds,
                    "active_seconds": active_seconds,
                }
            )
            + "\n"
        )
        self._previous_step_end_ns = ended_ns
        self._started_ns = None
        if (
            self.stop_after_steps is not None
            and state.global_step >= self.stop_after_steps
        ):
            control.should_training_stop = True
            control.should_epoch_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


    
def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    if os.environ.get("P40_DEVICE_MAP") == "auto":  # [p40] split the policy model over the visible GPUs
        _ngpu = torch.cuda.device_count()
        _mem = {i: os.environ.get("P40_MAX_MEM", "21GiB") for i in range(_ngpu)}
        _mem[0] = os.environ.get("P40_MAX_MEM_GPU0", "12GiB")  # reward models (NLI, CLIP) also load on cuda:0
        bnb_model_from_pretrained_args.update(dict(device_map="auto", max_memory=_mem))
    
    if "Qwen" in model_args.model_name_or_path:
        if training_args.bits in [4, 8]:
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["vision"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                )
            ))
    else:
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

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    processor, tokenizer = None, None

    if "Qwen" in model_args.model_name_or_path:
        model, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif "onevision" in model_args.model_name_or_path.lower():
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    elif "idefics" in model_args.model_name_or_path.lower():
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    else:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)

        if tokenizer is not None:
            tokenizer.padding_side = 'right'
    
    # for n,p in model.named_parameters():
    #     p.requires_grad=True

    train_datalists, datasets_name = get_datalists(training_args, shot=training_args.shot, dataset_name=training_args.dataset_name, scenario_num=training_args.scenario)
    
    # select functions
    set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}

    global_state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    global_state_dict.update(non_lora_state_dict)

    # [p40] P40_INIT_STATE=<server_model_round*.pth>: continue from a saved checkpoint (e.g. the 500-step one).
    # It is loaded into global_state_dict, which round 0 copies into the model via load_state_dict.
    init_state_path = os.environ.get("P40_INIT_STATE")
    if init_state_path:
        init_state = torch.load(init_state_path, map_location="cpu")
        missing = [k for k in global_state_dict if k not in init_state]
        unexpected = [k for k in init_state if k not in global_state_dict]
        if missing or unexpected:
            raise RuntimeError(f"P40_INIT_STATE key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        for k in global_state_dict:
            global_state_dict[k] = init_state[k].to(global_state_dict[k].dtype)
        print(f"[p40] loaded {len(init_state)} tensors from {init_state_path} "
              f"(sum of abs = {sum(float(t.float().abs().sum()) for t in init_state.values()):.6e})")
        del init_state

    current_task_vectors = None
    
    local_state_dict_list = [copy.deepcopy(global_state_dict)]
    old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
    local_state_dict_keys = local_state_dict_list[0].keys()
    extra_state_dict_dict = set_state_dict(model, global_state_dict, local_state_dict_list, training_args)

    training_loss = [[]]
    
    print("global_state_dict", list(global_state_dict.keys()))
    
    # start federated learning
    start_time = time.time()
    
    memory = [[]]
    memory_size = training_args.memory_size


    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    
    total_rounds = training_args.num_rounds * training_args.num_tasks
    last_task_id = [-1]
    
    lr_step = (init_lr - final_lr)/total_rounds
    mm_lr_step = (mm_init_lr - mm_final_lr)/total_rounds

    cul_sub_dataset = []
    info_per_round = {}
    selected_info_per_round = {}
    selected_info_round_per_round = {}
    selected_info_ind_per_round = {}
    task_samples_num = []
    prev_all_inputs = None
    prev_all_input_inds = None
    
    klasses = np.array([])
    memory_count = [np.array([])]
    datalist = []
    datalist_round = []
    datalist_prob = []
    ema_pool = []
    
    memory_info_list = []
    
    # updated_loss = []
    # retrieved_index = []
    for curr_round in range(total_rounds):
        old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
        # clients turn
        cids = np.arange(1).tolist()
        num_selection = 1
        selected_ids = sorted(random.sample(cids, num_selection)) 
        if training_args.local_rank == 0 or training_args.local_rank == -1: 
            logger.info(f"Round {curr_round} | selected_ids: {selected_ids}\n")
        
        # selected_ids = cids
        training_args.learning_rate = init_lr - lr_step*curr_round
        training_args.mm_projector_lr = mm_init_lr - mm_lr_step*curr_round
        if curr_round > 0 and training_args.is_wsd:
            training_args.warmup_ratio = 0
            training_args.warmup_steps = 0
        for r_i, idx in enumerate(range(num_selection)):
            model.config.use_cache = False
            torch.cuda.empty_cache()
            train_id = selected_ids[idx]     
            
            ##### simulate online memory insertion & get_batch ####
            sub_dataset = train_datalists[train_id][curr_round]['datalist']
            if training_args.seed > 1:
                random.shuffle(sub_dataset)
            task_sub_dataset_length = len(sub_dataset)
            num_iterations = train_datalists[train_id][curr_round]['num_iter']
            
            if curr_round > 0:
                cul_task_num = task_samples_num[-1]+len(sub_dataset)
                task_samples_num.append(cul_task_num)
            else:
                task_samples_num.append(len(sub_dataset)-1)
                

            logger.info(f"original datalist {len(sub_dataset)}")
            
            model.zero_grad()
            
            
            extra_state_dict_dict['client_id'] = train_id
            extra_state_dict_dict['curr_round'] = curr_round
            
            load_state_dict(model, global_state_dict, old_local_state_dict_list, train_id, training_args, extra_state_dict_dict)
            print('model loading done')
            
                    
                
            iteration = 0
            datalist = []
            datalist_round = []
            if not training_args.is_streamonly:
                # memory-only
                memory[train_id] = sub_dataset
                for i, sample in enumerate(sub_dataset):
                    iteration += num_iterations
                    if iteration >= 1:
                        for _ in range(int(iteration)):
                            
                            batch = random.sample(memory[train_id], k=min(len(memory[train_id]), total_batchsize))
                            mul = (total_batchsize//len(batch)) + 1
                            batch = (batch*mul)[:total_batchsize]
                            
                            datalist.extend(batch[:])
                            iteration -= 1
                            for batch_data in batch:

                                index = memory[train_id].index(batch_data)
                                position = bisect.bisect_left(task_samples_num, index)
                                datalist_round.append(position)
                                
                
                
            else:
                # stream-only
                # datalist = sub_dataset[:num_iterations*total_batchsize]
                datalist = sub_dataset
            
            if training_args.resume_train_round > 0 and curr_round+1<training_args.resume_train_round:
                continue

            train_dataset = make_inputs_dataloader(
                datalist=datalist,
                return_image="path",        # return PIL image objects
                model=model_args.model_name_or_path.lower(),
                diva_enabled=(training_args.reward_type == "diva"),
                diva_variant_num=training_args.diva_variant_num,
                diva_initial_difficulty=training_args.diva_initial_difficulty,
                diva_min_difficulty=training_args.diva_min_difficulty,
                diva_max_difficulty=training_args.diva_max_difficulty,
            )


            if "Qwen" in model_args.model_name_or_path and curr_round==0:
                tokenizer = processor.tokenizer
            
            logger.info(f"check datalist {len(datalist)}")
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train client {train_id} | num samples {len(datalist)}')

                
            trainer = create_trainer(model=model, args=training_args, processor=processor, tokenizer=tokenizer, train_dataset=train_dataset, model_name=model_args.model_name_or_path)
            step_timing_path = os.environ.get("STEP_TIMING_PATH")
            if step_timing_path:
                stop_after_steps = os.environ.get("STOP_AFTER_STEPS")
                trainer.add_callback(
                    WallClockStepTimingCallback(
                        step_timing_path,
                        stop_after_steps=(
                            int(stop_after_steps) if stop_after_steps else None
                        ),
                    )
                )
            
            # to prevent multiple 'input_ids'
            if "otter" in model_args.model_name_or_path.lower():
                if training_args.local_rank == 0 or training_args.local_rank == -1: 
                    logger.info("Attempting to patch Otter model...")

                real_model = model
                
                search_limit = 100 
                count = 0
                
                while (hasattr(real_model, 'base_model') or hasattr(real_model, 'module')) and count < search_limit:
                    if hasattr(real_model, 'base_model'):
                        real_model = real_model.base_model
                    elif hasattr(real_model, 'module'):
                        real_model = real_model.module
                    count += 1
                
                if count >= search_limit:
                    logger.warning("Model nesting too deep or circular. Using current layer.")

                TargetClass = real_model.__class__
                
                if training_args.local_rank == 0 or training_args.local_rank == -1: 
                    logger.info(f"Detected Otter model. Patching CLASS: {TargetClass.__name__}")

                if not hasattr(TargetClass, '_original_forward_backup'):
                    TargetClass._original_forward_backup = TargetClass.forward

                def patched_forward_class_level(self, *args, **kwargs):
                    if 'input_ids' in kwargs:
                        del kwargs['input_ids']
                    return TargetClass._original_forward_backup(self, *args, **kwargs)

                TargetClass.forward = patched_forward_class_level

            trainer.train()
            trainer.save_model(training_args.output_dir)
            
            # save local model
            # output_dir = os.path.join(training_args.state_dir, f"{train_id}_client_model_round{curr_round+1}.pth")
            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                state_dict.update(non_lora_state_dict)
            else:
                state_dict = {k: t.detach().cpu().clone() for k, t in model.named_parameters() if t.requires_grad}
            
            local_state_dict_list[train_id] = copy.deepcopy(state_dict)
            print("state_dict", list(state_dict.keys()))

            k_to_del = []
            for k in state_dict.keys():
                if k not in local_state_dict_keys:
                    k_to_del.append(k)
            for k in k_to_del:
                print("del", k)
                del state_dict[k]
            # if (training_args.local_rank == 0 or training_args.local_rank == -1):
            #     torch.save(state_dict, output_dir)
            
            local_state_dict = getattr(trainer, 'global_weight', None)
            if local_state_dict is not None:
                local_state_dict_list[train_id] = copy.deepcopy(local_state_dict)
            
            if getattr(trainer, "deepspeed", None) is not None:  # [p40] no DeepSpeed when the model is split by device_map
                trainer.deepspeed.empty_partition_cache()
            trainer.accelerator.free_memory()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"done Round {curr_round} client {train_id} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
    

        if training_args.resume_train_round > 0 and curr_round+1<training_args.resume_train_round:
            continue
        
        aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **extra_state_dict_dict)
        print("global", global_state_dict)
        # Save server model
        # if (training_args.local_rank == 0 or training_args.local_rank == -1): 
        torch.save(global_state_dict, os.path.join(training_args.state_dir, f"server_model_round{curr_round}.pth"))
        
        if getattr(trainer, "deepspeed", None) is not None:  # [p40] no DeepSpeed when the model is split by device_map
            trainer.deepspeed.empty_partition_cache()
        trainer.accelerator.free_memory()
        del trainer
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("total done\n")

def make_qwen_supervised_data_module(train_data, tokenizer,
                                data_args, model_name, curr_round) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if "onevision-1.5" in model_name.lower():
        train_dataset = OneVision1_5_LazySupervisedDataset(train_data, tokenizer, data_args)
        data_collator = OneVision1_5_DataCollatorForSupervisedDataset(tokenizer=tokenizer.tokenizer)
    elif "qwen3" in model_name.lower():
        train_dataset = QwenVL3_LazySupervisedDataset(train_data, tokenizer, data_args)
        data_collator = QwenVL3_DataCollatorForSupervisedDataset(tokenizer=tokenizer.tokenizer)
    elif "qwen" in model_name.lower():
        train_dataset = Qwen_LazySupervisedDataset(train_data, tokenizer, data_args, model_id=model_name)
        data_collator = Qwen_DataCollatorForSupervisedDataset(tokenizer=tokenizer.tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
def make_supervised_data_module(train_data, tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_name, curr_round, config=None, processor=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if "internvl3_5" in model_name.lower():
        train_dataset = Intern3_5_LazySupervisedDataset(train_data, tokenizer, data_args, config=config)
        data_collator = Intern3_5_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif "intern" in model_name.lower():
        train_dataset = Intern_LazySupervisedDataset(train_data, tokenizer, data_args, config=config)
        data_collator = Intern_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif "onevision-qwen" in model_name.lower():
        train_dataset = Onevision_LazySupervisedDataset(train_data, tokenizer, data_args)
        data_collator = Onevision_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif "idefics" in model_name.lower():
        train_dataset = Idefics_LazySupervisedDataset(train_data, tokenizer, data_args, processor=processor)
        data_collator = Idefics_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif "otter" in model_name.lower():
        train_dataset = Otter_LazySupervisedDataset(train_data, tokenizer, data_args)
        data_collator = Otter_DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        train_dataset = LazySupervisedDataset(train_data, tokenizer, data_args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def get_datalists(args, shot, dataset_name, scenario_num):
    with open(f"./scenarios/{shot}/{dataset_name}/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)

    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds
    
    datasets_name = []

    for train_data in scenario:
        train_id = train_data['client_id']
        train_datalist = []
        test_datalist = []
        eval_cnt = 0
        for task_id, data in enumerate(train_data['datasets']):
            datasets_name.append(data['dataset'])
            with open(f"./datalists/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            # random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            num_iter = max_iterations #max(int(max_iterations*samplenum_per_rounds/2000), 2) # 10000 / 5 = 2000
            for i in range(rounds_per_task):
                train_datalist.append(
                    {'datalist':datalist,
                     'num_iter': num_iter,
                     'task_id': task_id})
            
            train_datalists[train_id] = train_datalist

    return train_datalists, datasets_name

if __name__ == "__main__":
    main()
