import inspect
import os, sys, math, time, shutil, warnings
from typing import Optional, Dict, Union, Any, List
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.utils.data import RandomSampler
from packaging import version
import time

from transformers import Trainer
from transformers.utils import logging
from transformers.trainer import (
    TRAINER_STATE_NAME,
    SCHEDULER_NAME,
    is_sagemaker_mp_enabled,
    _is_peft_model,
    is_torch_xla_available,
    is_accelerate_available,
    is_deepspeed_available,
)
from transformers.trainer_utils import TrainOutput, has_length, speed_metrics
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler, reissue_pt_warnings
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers import Trainer
from transformers.trainer_callback import TrainerState
from transformers.training_args import ParallelMode

# your base trainer
from models.llava.llava_trainer import LLaVATrainer

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedType

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler
        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100


def fedavg_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # unchanged: your original behavior
    with torch.no_grad():
        model.load_state_dict(global_state_dict, strict=False)


def fedavg_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    for key in global_state_dict.keys():
        global_state_dict[key] = sum([local_state_dict_list[client][key] / num_selection for client in selected_ids])


def mimic_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict, return_task_vectors=False):
    trainer = LLaVATrainerFEDAVG(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        client_id=extra_state_dict_dict["client_id"],
        curr_round=extra_state_dict_dict["curr_round"],
        task_vector=extra_state_dict_dict.get("task_vector", None),
        fisher_old=extra_state_dict_dict.get("fisher_old", None),
        fisher_freq=extra_state_dict_dict.get("fisher_freq", 5),
        return_embedding=getattr(training_args, "embedding", None),
        return_task_vectors=return_task_vectors,
        eval_period=getattr(training_args, "eval_period", 8000),
        **data_module,
    )
    return trainer


class LLaVATrainerFEDAVG(LLaVATrainer):
    def __init__(
        self,
        client_id,
        curr_round,
        task_vector=None,
        fisher_old=None,
        fisher_freq=5,
        return_embedding=None,
        return_task_vectors=False,
        eval_period=8000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client_id = client_id
        self.curr_round = curr_round

        self.task_vector = task_vector.cuda() if task_vector is not None else None
        self.fisher_old = fisher_old
        self.fisher_cur = 0
        self.fisher_cnt = 0
        self.fisher_freq = fisher_freq

        self.return_embedding = return_embedding
        self.return_task_vectors = return_task_vectors

        self.eval_period = eval_period
        self.bs = 1  # you had this

    # ------------------------------------------------------------
    # Mask utils
    # ------------------------------------------------------------
    def _make_joint_mask(
        self,
        prefix_mask: Union[torch.Tensor, Dict[str, torch.Tensor]],
        shift_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns a boolean mask suitable for selecting tokens for alignment.

        Supported inputs:
          - prefix_mask: BoolTensor [B,T] OR dict containing "answer_mask"/"joint_mask"/"label_mask"
          - shift_mask : BoolTensor [B,T] (optional)
        Behavior:
          - If prefix_mask is dict: prefer "answer_mask" else "joint_mask" else "label_mask"
          - If shift_mask provided: returns BOTH masks in a tuple?  ❌ Not allowed by your usage.
            Instead: return prefix_mask only (for prefix sequence). For shift sequence, compute separately.
        """
        if isinstance(prefix_mask, dict):
            for k in ("answer_mask", "joint_mask", "label_mask", "mask"):
                if k in prefix_mask:
                    prefix_mask = prefix_mask[k]
                    break

        if not torch.is_tensor(prefix_mask):
            raise TypeError("prefix_mask must be a torch.Tensor or dict of tensors")

        prefix_mask = prefix_mask.bool()

        # IMPORTANT:
        # You cannot mix full-seq and query-seq masks into one mask tensor because they have different lengths.
        # So this returns only the mask for the sequence you pass in.
        return prefix_mask

    def _extract_masked_hidden(self, hidden_states, mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: tuple(list) of [B,T,D] (HF hidden_states)
        mask: BoolTensor [B,T]
        returns: [L, N, D]
        """
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)

        mask = mask.bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        hs = torch.stack(list(hidden_states), dim=0)  # [L,B,T,D]
        L, B, T, D = hs.shape
        hs = hs.reshape(L, B * T, D)

        flat_mask = mask.reshape(B * T)
        return hs[:, flat_mask, :]

    # ------------------------------------------------------------
    # Alignment losses
    # ------------------------------------------------------------
    def logits_kl_loss(self, shift_logits, prefix_logits, shift_mask, prefix_mask):
        """
        shift_logits: [B,T,V] for query-only
        prefix_logits: [B,T,V] for full-context
        shift_mask/prefix_mask: BoolTensor [B,T]
        """
        shift_mask = shift_mask.bool()
        prefix_mask = prefix_mask.bool()

        shift_sel = shift_logits[shift_mask]   # [N,V]
        prefix_sel = prefix_logits[prefix_mask]

        if shift_sel.shape[0] != prefix_sel.shape[0]:
            n = min(shift_sel.shape[0], prefix_sel.shape[0])
            shift_sel = shift_sel[:n]
            prefix_sel = prefix_sel[:n]

        return F.kl_div(
            shift_sel.log_softmax(dim=-1),
            prefix_sel.softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )

    def layer_wise_loss(self, shift_hidden_states, prefix_hidden_states, shift_mask, prefix_mask, mode="mse"):
        """
        Align hidden states tokenwise.
        Requires shift_mask & prefix_mask computed on their own sequences.
        """
        shift = self._extract_masked_hidden(shift_hidden_states, shift_mask)    # [L,Ns,D]
        prefix = self._extract_masked_hidden(prefix_hidden_states, prefix_mask) # [L,Np,D]

        # align lengths
        n = min(shift.shape[1], prefix.shape[1])
        shift = shift[:, :n, :]
        prefix = prefix[:, :n, :]

        if mode == "mse":
            return F.mse_loss(shift, prefix, reduction="mean")
        elif mode == "cos":
            shift_n = F.normalize(shift, dim=-1)
            prefix_n = F.normalize(prefix, dim=-1)
            cos = (shift_n * prefix_n).sum(dim=-1)  # [L,N]
            return (1.0 - cos.mean())
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------
    # Label mask
    # ------------------------------------------------------------
    def generate_label_mask(self, inputs: Dict[str, torch.Tensor], num_separator: int, keep_bos: bool = False):
        """
        ✅ Safe behavior:
          - If num_separator <= 0: return (labels != IGNORE_INDEX)
          - Else: use PAD separators like your old Code2 logic
        """
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("inputs must contain 'labels' for generate_label_mask()")

        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        pad_id = self.tokenizer.pad_token_id
        non_pad_mask = input_ids.ne(pad_id)

        # ✅ If you don't want separator logic, use labels directly (most robust)
        if num_separator is None or num_separator <= 0:
            return labels.ne(IGNORE_INDEX) & non_pad_mask

        B, T = input_ids.shape
        label_mask = torch.zeros((B, T), device=input_ids.device, dtype=torch.bool)

        # bos position = first non-pad
        bos_position = non_pad_mask.long().argmax(dim=1)

        for i in range(B):
            pad_positions = (input_ids[i] == pad_id).nonzero(as_tuple=False).squeeze(-1)

            # If left padded, ignore pad tokens before bos
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                pad_positions = pad_positions[pad_positions > bos_position[i]]

            if pad_positions.numel() < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens ({pad_positions.numel()}) than num_separator ({num_separator})"
                )

            sep_pos = pad_positions[num_separator - 1].item()
            label_mask[i, sep_pos + 1 :] = True

        label_mask = label_mask & non_pad_mask

        if keep_bos:
            label_mask[torch.arange(B, device=input_ids.device), bos_position] = True

        return label_mask

    # ------------------------------------------------------------
    # Core training_step (fixes your dict(query_inputs) crash)
    # ------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs, output_hidden_states=True)

        if self.args.past_index >= 0: 
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    
    def training_step(self, model, query_inputs, full_inputs, num_items_in_batch=None, return_outputs=False):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # --- (A) normalize batch structure ---
        # Your dataloader sometimes returns: (full_dict, query_dict)
        # and sometimes returns: dict
        if not isinstance(query_inputs, dict):
            # common bad case: query_inputs is a tensor/tuple/list -> you got:
            # ValueError: dictionary update sequence element #0 has length 5; 2 is required
            raise TypeError(
                f"query_inputs must be a dict. Got type={type(query_inputs)}. "
                f"Check collator/dataloader return format."
            )
        if not isinstance(full_inputs, dict):
            raise TypeError(
                f"full_inputs must be a dict. Got type={type(full_inputs)}. "
                f"Check collator/dataloader return format."
            )

        # --- (1) teacher / prefix ---
        with torch.no_grad():
            prefix_out = model(**full_inputs, output_hidden_states=True)
        prefix_logits = prefix_out["logits"]
        prefix_hs = prefix_out["hidden_states"]
        # print("prefix_hs", prefix_hs)
        # print("query_inputs labels",query_inputs["labels"])
        # --- (2) student / shift ---
        # Ensure labels exist if you want CE; easiest: use labels from dataset (preferred)
        # If your query_inputs doesn't have labels, you can do next-token objective by copying input_ids
        if "labels" not in query_inputs or query_inputs["labels"] is None:
            query_inputs = dict(query_inputs)  # safe now
            query_inputs["labels"] = query_inputs["input_ids"]

        # with self.compute_loss_context_manager():
        #     shift_out = model(**query_inputs, output_hidden_states=True)
        with self.compute_loss_context_manager():
            ce, shift_out = self.compute_loss(model, query_inputs, return_outputs=True)

        shift_logits = shift_out["logits"]
        shift_hs = shift_out["hidden_states"]

        loss_total = 0.0
        loss_dict = {}

        # --- (3) optional LM loss ---
        if ce is None:
            # some models only return loss if labels provided
            ce = torch.tensor(0.0, device=shift_logits.device)
        loss_dict["ce_loss"] = ce
        loss_total = loss_total + (0.5 * ce)
        # print("ce", ce)

        # --- (4) masks ---
        # ✅ robust, not separator-dependent:
        prefix_mask = full_inputs["labels"].ne(IGNORE_INDEX)
        shift_mask = query_inputs["labels"].ne(IGNORE_INDEX)

        # --- (5) hidden alignment ---
        mode = "mse"
        lw = self.layer_wise_loss(
            shift_hs, prefix_hs,
            shift_mask=shift_mask,
            prefix_mask=prefix_mask,
            mode=mode,
        )
        loss_dict[f"layer_{mode}_loss"] = lw
        loss_total = loss_total + lw
        
        # print("lw", lw)
        
        # # --- (6) logits KL ---
        # kl = self.logits_kl_loss(shift_logits, prefix_logits, shift_mask, prefix_mask)
        # loss_dict["logits_kl_loss"] = kl
        # loss_total = loss_total + (getattr(self.args, "align_loss_weight", 1.0) * kl)

        # --- backward ---
        if self.args.n_gpu > 1:
            loss_total = loss_total.mean()

        self.accelerator.backward(loss_total)

        return (loss_total.detach(), loss_dict) if return_outputs else loss_total.detach()

    # ------------------------------------------------------------
    # Inner training loop: includes the missing epochs_trained line
    # ------------------------------------------------------------
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        '''
        most lines are original transformer trainer code
        lines surrounded with
        ##############################################################################################################
        are added lines
        '''
        start_time = time.time()
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
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size * self.bs
        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps)
            # num_update_steps_per_epoch = len_dataloader // (args.gradient_accumulation_steps*self.bs)
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
            tr_loss_step, output = deepspeed_init(self, num_training_steps=max_steps)

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
        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

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
                    self.model, self.optimizer, self.lr_scheduler)
        if model is not self.model:
            self.model_wrapped = model

        # ---- resume logic (THIS is where your missing line lives) ----
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))

            # ✅ MISSING LINE (now included)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch

            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps * self.bs
            else:
                steps_trained_in_current_epoch = 0

            logger.info(f"Continuing training from checkpoint: global_step={self.state.global_step}, epochs_trained={epochs_trained}")

        # ---- logging header ----
        logger.info("***** Running training *****")
        if len_dataloader is not None:
            num_examples = self.num_examples(train_dataloader)
            logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = float(epochs_trained)
        start_time = time.time()
        
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0, device=args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # ---- skip batches if needed ----
        for epoch in range(epochs_trained, int(num_train_epochs) if num_train_epochs != sys.maxsize else epochs_trained + 10**9):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            if not args.ignore_data_skip and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_trained_in_current_epoch = 0

            for step, batch in enumerate(epoch_iterator):
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                # ✅ handle mimic dataloader return:
                # either dict OR (full_dict, query_dict)
                if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], dict) and isinstance(batch[1], dict):
                    full_inputs, query_inputs = batch[0], batch[1]
                    is_mimic = True
                else:
                    # non-mimic: use same inputs as both
                    full_inputs = batch if isinstance(batch, dict) else batch[0]
                    query_inputs = batch if isinstance(batch, dict) else batch[0]
                    is_mimic = False

                # prepare inputs (HF helper)
                full_inputs = self._prepare_inputs(full_inputs)
                query_inputs = self._prepare_inputs(query_inputs)

                with self.accelerator.accumulate(model):
                    loss_step, _loss_dict = self.training_step(model, query_inputs, full_inputs, return_outputs=True)
                    tr_loss += loss_step
    
                    if self.accelerator.sync_gradients:
                        grad_norm = None
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )
                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.optimizer.step()
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()
                        model.zero_grad()

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / max(1, (len_dataloader or 1))
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        
                        # # save client model periodically (your behavior)
                        # if (step + 1) % self.eval_period == 0:
                        #     out_path = os.path.join(self.args.state_dir, f"{self.client_id}_client_model_round{self.curr_round+1}_itr{step+1}.pth")
                        #     if (self.args.local_rank == 0 or self.args.local_rank == -1):
                        #         state_dict = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                        #         torch.save(state_dict, out_path)

                        # log/save/eval
                        maybe_log_params = inspect.signature(
                            self._maybe_log_save_evaluate
                        ).parameters
                        maybe_log_args = (
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                        )
                        if "start_time" in maybe_log_params:
                            self._maybe_log_save_evaluate(
                                *maybe_log_args, start_time=start_time
                            )
                        else:
                            self._maybe_log_save_evaluate(*maybe_log_args)
                        
                        
                        if self.state.global_step >= max_steps:
                            break

            if self.state.global_step >= max_steps:
                break

        # ---- finalize ----
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 1)
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics("train", start_time, num_samples=None, num_steps=self.state.max_steps, num_tokens=None)
        metrics["train_loss"] = train_loss
        self.log(metrics)

        return TrainOutput(self.state.global_step, train_loss, metrics)
