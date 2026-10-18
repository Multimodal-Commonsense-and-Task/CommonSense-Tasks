import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler
from packaging import version
from torch import nn
from utils.train_utils import load_deepspeed
from models.llava.llava_trainer import LLaVATrainer
# from transformers.utils import logging
import logging.config
import sys, os, time, shutil
import math
from typing import Optional, Dict, Union, Any
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler, reissue_pt_warnings
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers import Trainer
import bitsandbytes
from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    _is_peft_model, 
    TRAINER_STATE_NAME,
    is_torch_xla_available,
    is_accelerate_available,
    is_deepspeed_available,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    SCHEDULER_NAME
)
import warnings
from transformers.integrations import hp_params
from transformers.trainer_callback import TrainerState
from transformers.training_args import ParallelMode

import numpy as np
from torch.nn.functional import normalize
import h5py
from typing import List
import copy

from selection_methods.fedavg import LLaVATrainerFEDAVG
from collections import defaultdict
import math
from scipy.special import softmax

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]
    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

# logger = logging.get_logger(__name__)
logging.config.fileConfig("./configuration/logging.conf")
logger = logging.getLogger()

_supported_layers = ['Linear']


def divbs_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, curr_ema=None, curr_ema_var=0, datalist_prob=None, ema_data_module=None, ema_pool=None, return_task_vectors=False):
    trainer = LLaVATrainerDIVBS(model=model,
        ema_ratio=training_args.ema_ratio,
        mode=training_args.mode,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        task_vector=extra_state_dict_dict['task_vector'] if 'task_vector' in extra_state_dict_dict else None,
        fisher_old=extra_state_dict_dict['fisher_old'] if 'fisher_old' in extra_state_dict_dict else None,
        fisher_freq=extra_state_dict_dict['fisher_freq'] if 'fisher_freq' in extra_state_dict_dict else 5,
        return_embedding=training_args.embedding,
        return_task_vectors=return_task_vectors,
        selection_method=training_args.selection_method,
        kmeans=training_args.k_means,
        curr_ema=curr_ema,
        curr_ema_var=curr_ema_var,
        min_z_score_threshold=training_args.min_z_score_threshold, 
        max_z_score_threshold=training_args.max_z_score_threshold,
        start_ema_update_step=training_args.start_ema_update_step,
        sample_ratio=training_args.sample_ratio,
        datalist_prob=datalist_prob,
        ema_update_mode=training_args.ema_update_mode,
        softmax_update_temp=training_args.softmax_update_temp,
        ema_data_module=ema_data_module,
        ema_average=training_args.ema_average,
        ema_pool=ema_pool,
        ema_pool_size=training_args.ema_pool_size,
        eval_period=training_args.eval_period,
        **data_module,
        )
    return trainer



class LLaVATrainerDIVBS(LLaVATrainerFEDAVG):
    def __init__(self, ema_ratio,mode,client_id, curr_round, task_vector=None, fisher_old=None, fisher_freq=5, return_embedding=None, return_task_vectors=False, selection_method=None, kmeans=None, curr_ema=None, curr_ema_var=0, min_z_score_threshold=0, max_z_score_threshold=0, start_ema_update_step=0, sample_ratio=0, datalist_prob=None, ema_update_mode=None, softmax_update_temp=0,ema_data_module=None, ema_average=False, ema_pool=None, ema_pool_size=0, eval_period=8000,**kwargs):
        super(LLaVATrainerDIVBS, self).__init__(client_id=client_id, curr_round=curr_round, task_vector=task_vector, fisher_old=fisher_old, fisher_freq=fisher_freq, return_embedding=return_embedding, return_task_vectors=return_task_vectors, selection_method=selection_method, kmeans=kmeans, **kwargs)
        self.ema_ratio = ema_ratio
        self.mode = mode
        self.curr_ema = curr_ema
        self.curr_ema_var = curr_ema_var
        self.round_info = []
        self.selected_info = []
        self.selected_info_ind = []
        # self.selected_info_round = []
        # if self.mode == "fisherBudgetOrg":
        #     self.computation_per_sample = torch.tensor(8962304901120.0)
        # elif self.mode == "entropyBudgetOrg":
        #     self.computation_per_sample = torch.tensor(7865129500672.0)
        
        self.target_update_parameters = []
        target_update_parameters = self.model_trainable_block()
        for blocks in target_update_parameters:
            self.target_update_parameters.extend(blocks)
        # self.block_name = self.generate_llava_blocks()
        # self.num_blocks = len(self.block_names) - 1
        # print("self.num_blocks", self.num_blocks)
        # self.cumulative_fisher = []
        self.min_z_score_threshold = min_z_score_threshold
        self.max_z_score_threshold = max_z_score_threshold
        self.start_ema_update_step = start_ema_update_step
        self.sample_ratio = sample_ratio
        
        self.ema_list = []
        self.datalist_prob = datalist_prob
        self.ema_update_mode=ema_update_mode
        self.softmax_update_temp=softmax_update_temp
        
        self.ema_average = ema_average
        self.ema_pool = ema_pool
        self.ema_pool_size = ema_pool_size
        self.eval_period = eval_period
    
    
    def greedy_selection(self, grad_mean, grad):
        number_to_select = self.sample_size_perbatch
        residual = grad_mean.unsqueeze(-1) if grad_mean.dim() == 1 else grad_mean
        index_selected = []
        D = grad.t()
        selected_element = []
        for i in range(number_to_select):
            correlations = torch.abs(torch.matmul(D.t(), residual))
            if index_selected:
                correlations[index_selected] = 0
            if not torch.isfinite(correlations).all() or correlations.sum() <= 0:
                break
            try:
                idx = torch.multinomial(correlations.squeeze(), 1)
            except:
                break
            index_selected.append(idx.item())
            if len(selected_element) >0:
                selected_element_matrix = torch.cat(selected_element, dim=1) # n_features, n_selected
                D_selected = D[:, idx] - torch.matmul(selected_element_matrix,torch.matmul(selected_element_matrix.t(), D[:, idx]))
            else:
                D_selected = D[:, idx]
            norm = torch.norm(D_selected)
            if norm == 0:
                break
            D_selected = D_selected / norm
            selected_element.append(D_selected)

            
            residual = residual - torch.matmul(D_selected.t(), residual)* D_selected
        
        if len(selected_element) < number_to_select:
            num_random = number_to_select - len(selected_element)
            logger.info(f'number_to_select: {number_to_select}, len(selected_element): {len(selected_element)}, num_random: {num_random}')
            remaining_indices = list(set(range(grad.shape[0])) - set(index_selected))
            num_random = min(num_random, len(remaining_indices))
            random_indices = np.random.choice(remaining_indices, num_random, replace=False)
            index_selected.extend(random_indices)

        return index_selected
    
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        '''
        most lines are original transformer trainer code
        lines surrounded with
        ##############################################################################################################
        are added lines
        '''
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        # self.is_deepspeed_enabled = False
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        # total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size *self.bs
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size *self.bs
        self.sample_size_perbatch = int(total_train_batch_size*self.sample_ratio)
        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            # num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps*self.bs/self.sample_size_perbatch)
            # num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps*self.bs)
            num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False
        # use_accelerator_prepare = False
        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # use_accelerator_prepare = False

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        #############################################################################################################

        if self.args.save_optim and self.curr_round > 0:
            output_dir = f'model_states_{self.args.note}/client_{self.client_id}/'
            self._load_optimizer_and_scheduler(output_dir)
            
        ##############################################################################################################
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= (args.gradient_accumulation_steps*self.bs)
                # steps_trained_in_current_epoch *= (args.gradient_accumulation_steps)
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        large_zcore_ind = []
        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        org_weights = {}
        saved_org_grads = defaultdict(list)
        selected_inputs = []
        full_batch_info = []
        cumulative_sum = 0
        count = 0
        batch_inputs = []
        curr_batch_ind = 0
        all_reps = []
        batch_selected_ind = []
        selected_samples = []
        batch_info = []
        batch_inds = []
        batch_gradients = []
        batch_features = []
        selected_num_per_batch = int(total_train_batch_size*self.sample_ratio)
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * (args.gradient_accumulation_steps*self.bs)
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            # emb_memory_loc = f"embedding_{self.selection_method}_k{self.kmeans}.npy"
            accumulated_batches, accul_threshold = [], 5
            add_accul = True
            start = 0
            # with h5py.File(emb_memory_loc, 'w') as f:
            #     dataset = f.create_dataset(
            #         "embeddings", 
            #         shape=(len(self.train_dataset), 4213, 4096),  # Start with zero entries
            #         maxshape=(len(self.train_dataset), 4213, 4096),  # Allow unlimited rows
            #         dtype=np.float32,  # Use float32 to save space
            #         compression="gzip",  # Optional compression
            #         compression_opts=9,
            #         chunks=(20, 4213, 4096)
            #     )
            # for n, p in model.base_model.model.model.layers[-2].mlp.up_proj.lora_B.default.named_parameters():
            #     copied_param = copy.deepcopy(p.data)
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            )
                        ).item()
                
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % (args.gradient_accumulation_steps * self.bs) == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps *self.bs and (step + 1) == steps_in_epoch
                    )
                
                curr_inputs = copy.deepcopy(inputs)
                if not self.return_task_vectors:
                    # if step % (args.gradient_accumulation_steps*args.gradient_accumulation_steps) == 0:
                    # if step % self.bs == 0:
                    changed_parameters = []
                    for n, p in model.base_model.named_parameters():
                        if p.requires_grad == True:
                            changed_parameters.append(n)
                            p.requires_grad = False
                    for n, p in model.base_model.model.model.layers[-1].mlp.down_proj.base_layer.named_parameters():
                        p.requires_grad = True
                        #     org_weights[n] = p.clone().detach()


                    inputs = self._prepare_inputs(inputs)
                    output = self.model(**inputs, output_hidden_states=True).loss
                    # output.loss.backward()
                    # output = output.loss

                    for n, p in model.base_model.model.model.layers[-1].mlp.down_proj.base_layer.named_parameters():
                        # grad = p.grad.clone().detach()
                        grad = torch.autograd.grad(output, p, retain_graph=True)[0]
                        # batch_features.append(output["hidden_states"][-1])
                        # batch_gradient.append(torch.mean(grad, dim=0).to(torch.float32))
                        batch_gradients.append(grad.flatten())
                    
                    # for n, p in model.module.base_model.model.model.layers[-1].mlp.down_proj.base_layer.named_parameters():
                    #     # info = float(((p.grad.clone().cpu().detach())**2).sum().item())
                    #     info = ((p.grad.clone().cpu().detach().clamp(-1,1))**2).sum()
                    #     logger.info(f"step{step} {info.item()}")
                    #     # p.grad = torch.zeros_like(p).to(p.dtype)
                    #     self.round_info.append(info.item())#[:,0].view(-1))

                    # if step % (self.bs-1) == 0:
                    for n, p in model.base_model.model.model.layers[-1].mlp.down_proj.base_layer.named_parameters():
                        p.requires_grad = False
                    for n, p in model.base_model.named_parameters():
                        if n in changed_parameters:
                            p.requires_grad = True
                    
                    model.zero_grad()

                    batch_inds.append(step)
                    batch_inputs.append(curr_inputs)

                if (
                    total_batched_samples % (args.gradient_accumulation_steps*self.bs) == 0
                    # len(selected_samples) == args.gradient_accumulation_steps
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    
                    batch_gradient_tensor = torch.stack(batch_gradients)
                    batch_gradient_tensor = batch_gradient_tensor.view(batch_gradient_tensor.shape[0],-1)
                    batch_gradient_sum = batch_gradient_tensor.mean(dim=0)
                    
                    indices = self.greedy_selection(batch_gradient_sum, batch_gradient_tensor)
                    for ind in indices:
                        # self.selected_info.append(batch_gradient_tensor[ind].cpu())
                        self.selected_info_ind.append(batch_inds[ind])
                        selected_samples.append(batch_inputs[ind])
                    
                
                    
                    for sample in selected_samples:
                        if self.args.include_num_input_tokens_seen:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            input_device = sample[main_input_name].device
                            self.state.num_input_tokens_seen += torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(sample[main_input_name].numel(), device=input_device, dtype=torch.int64)
                                )
                            ).item()
                        
                        with self.accelerator.accumulate(model):
                            tr_loss_step, output = self.training_step(model,sample, return_outputs=True)
    
                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss += tr_loss_step

                        curr_batch_ind = 0
                        # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    
                    # wsd
                    if self.args.is_wsd == 'WSD' and math.ceil(self.state.epoch*steps_in_epoch) == math.ceil(self.args.decay_ratio*steps_in_epoch):
                        self.global_weight = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                
                    # # save client model
                    # if (step+1) % self.eval_period == 0:
                    #     output_dir = os.path.join(self.args.state_dir, f"{self.client_id}_client_model_round{self.curr_round+1}_itr{step+1}.pth")
                    #     state_dict = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                    #     if (self.args.local_rank == 0 or self.args.local_rank == -1):
                    #         torch.save(state_dict, output_dir)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval,0)
                 
                    
                    # self.accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
                    selected_samples = []
                    curr_batch_ind = 0
  
                    # compute fisher online
                    if step % self.fisher_freq == 0:
                        
                        model.zero_grad()
                        
                    batch_info = []
                    batch_inputs = []
                    batch_inds = []
                    batch_gradients = []
                    
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
                
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval,0)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        ##############################################################################################################
        if self.args.save_optim:
            output_dir = f'model_states_{self.args.note}/client_{self.client_id}/'
            self._save_optimizer_and_scheduler(output_dir)
        ##############################################################################################################
        
        # if self.return_embedding == "gradient" and self.return_task_vectors:
        #     self.fisher_old = ((self.fisher_cur/self.fisher_cnt) + self.fisher_old) / 2 if self.fisher_old is not None else (self.fisher_cur/self.fisher_cnt)
        #     self.task_vector = self.fisher_old.detach().cpu()
        # elif self.return_embedding == "representation" and self.return_task_vectors:
        #     all_reps = torch.cat(all_reps).numpy()
        #     self.task_vector = all_reps
        # all_reps = torch.cat(all_reps).numpy()
        # self.task_vectors = all_reps
        
        return TrainOutput(self.state.global_step, train_loss, metrics)
            
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None, return_outputs=False, no_grad_cal=False, num_sample=1
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, output = self.compute_loss(model, inputs, return_outputs=True)

        del inputs
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if not no_grad_cal:
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # loss /= torch.tensor(num_sample).cuda()
                self.accelerator.backward(loss)

        return (loss.detach(), output) if return_outputs else loss.detach()
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            from deepspeed.runtime.state_dict_factory import SDLoaderFactory
            from deepspeed.runtime.pipe.module import PipelineModule
            latest_tag = "latest"
            latest_path = os.path.join(checkpoint, latest_tag)
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip()
                    
            ckpt_list = self.model_wrapped._get_all_ckpt_names(checkpoint, tag)
            sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, checkpoint_engine=self.model_wrapped.checkpoint_engine)

            is_pipe_parallel = isinstance(self.model_wrapped.module, PipelineModule)

            mp_rank = 0 if self.model_wrapped.mpu is None else self.model_wrapped.mpu.get_model_parallel_rank()
            load_path, checkpoint_state, _ = sd_loader.load(self.model_wrapped.mp_world_size, mp_rank, is_pipe_parallel=is_pipe_parallel)
            self.model_wrapped.loaded_checkpoint_dp_world_size = checkpoint_state['dp_world_size']
            self.model_wrapped.loaded_checkpoint_mp_world_size = checkpoint_state['mp_world_size']

            zero_sd_list = self.model_wrapped._get_all_zero_checkpoints(checkpoint, tag)
            self.model_wrapped.optimizer.load_state_dict(state_dict_list=zero_sd_list,
                                       load_optimizer_states=True,
                                       load_from_fp32_weights=self.model_wrapped.zero_load_from_fp32_weights(),
                                       checkpoint_folder=None,
                                       load_serial=None)
            
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
            return

        else:
            super()._load_optimizer_and_scheduler(checkpoint)
