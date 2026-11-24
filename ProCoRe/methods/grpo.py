import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler
from packaging import version
from torch import nn
from utils.train_utils import load_deepspeed
from models.llava.llava_trainer import LLaVATrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
from trl.rewards import think_format_reward
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.trainer.grpo_trainer import (
    entropy_from_logits,
    get_high_entropy_mask,
    nanmax,
    nanmin,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_tensor_dict,
)
from transformers.utils import logging
import sys, os, time, shutil, itertools
import math
from typing import Optional, Dict, Union, Any, Callable
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
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
    SCHEDULER_NAME
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

import open_clip
import warnings
from transformers.integrations import hp_params
from transformers.trainer_callback import TrainerState
from transformers.training_args import ParallelMode

import numpy as np
from torch.nn.functional import normalize
import h5py
from typing import List
import random
import copy
import json
import datasets
from datasets import Dataset, IterableDataset
from evaluate import load
from PIL import Image
from utils.reward_functions import configure_proposition_decomposer, get_reward_func
from methods.vgpo import (
    compute_inter_trajectory_weights,
    compute_intra_trajectory_weights,
    compute_visual_accumulation,
    compute_visual_focus_scores,
)
from methods.grit import GRITPromptDataset
from methods.diva import (
    DIVARepeatSampler,
    compute_difficulty_updates,
    compute_diva_advantages,
    parse_difficulty_changes,
    parse_score_ranges,
    validate_difficulty_schedule,
)
from methods.pepo import (
    anneal_pepo_token_weights,
    compute_pepo_token_weights,
    compute_vision_scores,
    validate_pepo_config,
)

from collections import defaultdict
from deepspeed.utils import safe_get_full_grad, safe_set_full_grad
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

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

logger = logging.get_logger(__name__)

_supported_layers = ['Linear']

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

_VISUAL_CONTROL_TOKEN_STRINGS = (
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
)


def _resolve_visual_control_token_ids(processing_class, model) -> list[int]:
    """Resolve multimodal control tokens that must never be generated.

    These tokens are valid in a processor-built prompt only when paired with
    pixel/grid inputs. If the language model samples one in a completion, the
    subsequent policy/reference log-probability pass sees an extra image block
    with no corresponding image grid and cannot align the visual inputs.
    """
    token_ids = set()
    model_config = getattr(model, "config", None)
    for owner in (processing_class, model_config):
        if owner is None:
            continue
        for attribute in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            token_id = getattr(owner, attribute, None)
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.add(token_id)

    tokenizer = getattr(processing_class, "tokenizer", None)
    if tokenizer is None and hasattr(processing_class, "convert_tokens_to_ids"):
        tokenizer = processing_class
    if tokenizer is not None:
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        for token in _VISUAL_CONTROL_TOKEN_STRINGS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0:
                continue
            if (
                token_id == unk_token_id
                and token != getattr(tokenizer, "unk_token", None)
            ):
                continue
            token_ids.add(int(token_id))

    return sorted(token_ids)

# ------------------------------------------------------------------
# globals for hooks
# ------------------------------------------------------------------
_hooks_disabled = False
_enforce_fresh_backprop = True

# ------------------------------------------------------------------
# cache reward models so they are loaded only once
# ------------------------------------------------------------------
_REWARD_CACHE = {
    "nli_tokenizer": None,
    "nli_model": None,
    "bertscore": None,
    "vg_processor": None,
    "vg_tokenizer": None,
    "vg_model": None,
    "device": None,
}


class MultiImageProcessorProxy(ProcessorMixin):
    """Normalize TRL 0.20's single-image wrapper for multi-image samples."""

    attributes = []

    def __init__(self, processor):
        self._processor = processor
        self.tokenizer = processor.tokenizer

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __call__(self, *args, **kwargs):
        images = kwargs.get("images")
        if images is not None:
            normalized_images = []
            for sample_images in images:
                if (
                    isinstance(sample_images, (list, tuple))
                    and len(sample_images) == 1
                    and isinstance(sample_images[0], (list, tuple))
                ):
                    normalized_images.append(list(sample_images[0]))
                else:
                    normalized_images.append(sample_images)
            kwargs["images"] = normalized_images
        return self._processor(*args, **kwargs)


def fedavg_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    model_to_load = global_state_dict
    with torch.no_grad():
        model.load_state_dict(model_to_load, strict=False)  

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def grpo_create_trainer(model, args, processor, tokenizer, train_dataset, model_name):
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

    reward_weights = args.reward_weight
    grit_enabled = args.reward_type == "grit"
    diva_enabled = args.reward_type == "diva"
    pepo_enabled = args.reward_type == "pepo"
    fast_enabled = args.reward_type == "fast"

    if processor is None:
        processing_class = tokenizer
    else:
        processing_class = MultiImageProcessorProxy(processor)

    if grit_enabled:
        train_dataset = GRITPromptDataset(train_dataset)

    training_args = GRPOConfig(
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        **({} if os.environ.get("P40_GRPO_SETUP") == "1" else {"bf16": args.bf16}),
        per_device_train_batch_size=args.per_gpu_train_batch_size,  # Increased from 1
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Reduced from 4
        num_generations=args.grpo_num_gen,  # Reduced from 16
        # VLM visual-token sequences must not be truncated.
        max_prompt_length=None,
        max_completion_length=(
            args.grit_max_completion_length if grit_enabled else 256
        ),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        optim="adamw_torch",
        top_entropy_quantile=(
            1.0
            if grit_enabled or diva_enabled or pepo_enabled or fast_enabled
            else 0.2
        ),
        beta=(
            0.0
            if diva_enabled
            else (args.pepo_beta if pepo_enabled else 0.01)
        ),
        reward_weights=reward_weights,
        # [p40] P40_GRPO_SETUP=1: actually use gradient checkpointing, the requested precision and the
        # DeepSpeed config (ZeRO-2 + CPU optimizer offload). Off by default (original behaviour).
        **({"gradient_checkpointing": bool(args.gradient_checkpointing),
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "fp16": bool(args.fp16), "bf16": bool(args.bf16) and not bool(args.fp16),
            "deepspeed": args.deepspeed}
           if os.environ.get("P40_GRPO_SETUP") == "1" else {}),
        epsilon=(args.grit_clip_epsilon if grit_enabled else 0.2),
        loss_type=("grpo" if grit_enabled or fast_enabled else "bnpo"),
        # use_vllm=True,
    )
    
    if args.reward_type in {"vgv2", "procore"}:
        configure_proposition_decomposer(
            model=args.proposition_model,
            cache_path=args.proposition_cache_path,
            device=args.proposition_device,
        )
    if args.reward_type in {"caprl", "caprlpp"}:
        from utils.caprl_reward import configure_caprl_reward

        configure_caprl_reward(
            qa_source=args.caprl_qa_source,
            qa_model=args.caprl_qa_model,
            qa_cache_path=args.caprl_qa_cache_path,
            qa_num=args.caprl_qa_num,
            qa_all=args.caprl_qa_all,
            num_options=args.caprl_num_options,
            shuffle_qa=args.caprl_shuffle_qa,
            answer_model=args.caprl_answer_model,
            answer_device=args.caprl_answer_device,
            answer_batch_size=args.caprl_answer_batch_size,
            answer_temperature=args.caprl_answer_temperature,
            acc_scale=args.caprl_acc_scale,
            format_mode=args.caprl_format_mode,
            format_weight=args.caprl_format_weight,
            length_weight=args.caprl_length_weight,
            length_l1=args.caprl_length_l1,
            length_l2=args.caprl_length_l2,
            length_tokenizer=args.caprl_length_tokenizer,
            debug=args.caprl_debug,
        )
    if fast_enabled:
        from utils.fast_reward import configure_fast_reward

        configure_fast_reward(
            group_size=args.grpo_num_gen,
            correctness_threshold=args.fast_correctness_threshold,
            difficulty_percentile=args.fast_difficulty_percentile,
            format_weight=args.fast_format_weight,
            length_weight=args.fast_length_weight,
        )
    reward_fns = get_reward_func(args.reward_type)
    vgpo_enabled = args.reward_type == "vgpo"
    diva_config = {}
    if diva_enabled:
        diva_score_ranges = parse_score_ranges(args.diva_score_ranges)
        diva_difficulty_changes = parse_difficulty_changes(
            args.diva_difficulty_changes
        )
        validate_difficulty_schedule(
            diva_score_ranges, diva_difficulty_changes
        )
        diva_config = {
            "weight_mode": args.diva_weight_mode,
            "score_ranges": diva_score_ranges,
            "difficulty_changes": diva_difficulty_changes,
            "min_difficulty": args.diva_min_difficulty,
            "max_difficulty": args.diva_max_difficulty,
            "weighted_advantage_k": args.diva_weighted_advantage_k,
            "rrb_threshold": args.diva_rrb_threshold,
            "rrb_scale_factor": args.diva_rrb_scale_factor,
            "local_weight": args.diva_local_advantage_weight,
            "global_weight": args.diva_global_advantage_weight,
        }
    pepo_config = {}
    if pepo_enabled:
        validate_pepo_config(
            gate_alpha=args.pepo_gate_alpha,
            gate_temperature=args.pepo_gate_temperature,
            vision_score_type=args.pepo_vision_score_type,
        )
        pepo_config = {
            "gate_alpha": args.pepo_gate_alpha,
            "gate_temperature": args.pepo_gate_temperature,
            "use_vision_weights": args.pepo_use_vision_weights,
            "vision_score_type": args.pepo_vision_score_type,
            "image_token": args.pepo_image_token,
        }
    trainer = My_GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=processing_class,
        model_name=model_name,
        vgpo_enabled=vgpo_enabled,
        vgpo_config={
            "use_intra_trajectory_reweighting": args.vgpo_use_intra_trajectory_reweighting,
            "use_inter_trajectory_reweighting": args.vgpo_use_inter_trajectory_reweighting,
            "use_visual_compensation": args.vgpo_use_visual_compensation,
            "use_gated_visual_compensation": args.vgpo_use_gated_visual_compensation,
            "gated_visual_compensation_start_ratio": args.vgpo_gated_visual_compensation_start_ratio,
            "visual_compensation_strength": args.vgpo_visual_compensation_strength,
            "visual_attention_threshold": args.vgpo_visual_attention_threshold,
        },
        diva_enabled=diva_enabled,
        diva_config=diva_config,
        pepo_enabled=pepo_enabled,
        pepo_config=pepo_config,
    )

    return trainer

def fedavg_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    for key in global_state_dict.keys():
        global_state_dict[key] = sum([local_state_dict_list[client][key] / num_selection for client in selected_ids])
        

def _p40_resume_step():
    return int(os.environ.get("P40_RESUME_STEP") or 0)


class _P40SkipSampler(torch.utils.data.Sampler):
    """[p40] Drops the first `skip` indices of `base` (data already consumed before a resume)."""

    def __init__(self, base, skip):
        self.base, self.skip = base, skip

    def __iter__(self):
        return itertools.islice(iter(self.base), self.skip, None)

    def __len__(self):
        return len(self.base) - self.skip


class My_GRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        model_name: str = None,
        vgpo_enabled: bool = False,
        vgpo_config: Optional[dict[str, Any]] = None,
        diva_enabled: bool = False,
        diva_config: Optional[dict[str, Any]] = None,
        pepo_enabled: bool = False,
        pepo_config: Optional[dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.vgpo_enabled = vgpo_enabled
        self.vgpo_config = vgpo_config or {}
        self.diva_enabled = diva_enabled
        self.diva_config = diva_config or {}
        self.pepo_enabled = pepo_enabled
        self.pepo_config = pepo_config or {}
        self.pepo_use_vision_weights = bool(
            self.pepo_config.get("use_vision_weights", False)
        )
        self._diva_last_rewards = None
        self._diva_last_metadata = None
        super().__init__(model=model, reward_funcs=reward_funcs, args=args, train_dataset=train_dataset, processing_class=processing_class, reward_processing_classes=reward_processing_classes, callbacks=callbacks, optimizers=optimizers, peft_config=peft_config)
        self.suppressed_visual_token_ids = _resolve_visual_control_token_ids(
            self.processing_class, self.model
        )
        if self.suppressed_visual_token_ids and hasattr(self, "generation_config"):
            existing_suppressed_tokens = (
                self.generation_config.suppress_tokens or []
            )
            self.generation_config.suppress_tokens = sorted(
                set(existing_suppressed_tokens)
                | set(self.suppressed_visual_token_ids)
            )
            if self.accelerator.is_main_process:
                print(
                    "[GRPO] Suppressing generated visual control token IDs: "
                    f"{self.generation_config.suppress_tokens}"
                )
        if self.vgpo_enabled:
            if self.use_liger_loss:
                raise ValueError("VGPO is not compatible with TRL's fused Liger loss.")
            if self.accelerator.is_main_process:
                print(
                    "[VGPO] Enabled visual compensation and dual-grained "
                    "advantage reweighting."
                )
        if self.diva_enabled:
            if self.use_liger_loss:
                raise ValueError("DIVA-GRPO is not compatible with Liger loss.")
            if not hasattr(self.train_dataset, "variants_per_source"):
                raise ValueError(
                    "DIVA-GRPO requires the DIVA adaptive dataset wrapper."
                )
            if self.accelerator.is_main_process:
                print(
                    "[DIVA-GRPO] Using Rouge-L rewards, difficulty-adaptive "
                    "visual variants, and joint local/global advantages "
                    f"({self.diva_config['weight_mode']})."
                )
        if self.pepo_enabled:
            if self.use_liger_loss:
                raise ValueError("PEPO is not compatible with Liger loss.")
            self.pepo_image_token_id = self.image_token_id
            configured_image_token = self.pepo_config.get("image_token")
            if configured_image_token:
                tokenizer_or_processor = getattr(
                    self.processing_class,
                    "tokenizer",
                    self.processing_class,
                )
                self.pepo_image_token_id = (
                    tokenizer_or_processor.convert_tokens_to_ids(
                        configured_image_token
                    )
                )
            if (
                self.pepo_use_vision_weights
                and self.pepo_image_token_id is None
            ):
                raise ValueError(
                    "PEPO could not resolve the image token ID. Set "
                    "--pepo_image_token for this model."
                )
            if self.accelerator.is_main_process:
                print(
                    "[PEPO] Using Rouge-L rewards and perception-exploration "
                    "token weighting "
                    f"({self.pepo_config['vision_score_type']}, "
                    f"alpha={self.pepo_config['gate_alpha']}, "
                    f"temperature={self.pepo_config['gate_temperature']})."
                )

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # [p40] P40_RESUME_STEP=k continues a longer run's lr schedule at step k (fresh optimizer):
        # the schedule spans k + num_training_steps steps and starts at position k.
        offset = _p40_resume_step()
        if offset <= 0 or self.lr_scheduler is not None:
            return super().create_scheduler(num_training_steps, optimizer)
        from transformers import get_scheduler
        optimizer = self.optimizer if optimizer is None else optimizer
        total = num_training_steps + offset
        base = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.get_warmup_steps(total),
            num_training_steps=total,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [lambda s, f=f: f(s + offset) for f in base.lr_lambdas]
        )
        self._created_lr_scheduler = True
        logger.warning(
            f"[p40] lr schedule resumed at step {offset}/{total} "
            f"(warmup {self.args.get_warmup_steps(total)}), lr now {self.lr_scheduler.get_last_lr()}"
        )
        return self.lr_scheduler

    def _get_train_sampler(self, dataset=None):
        # [p40] P40_RESUME_STEP=k with P40_FULL_STEPS=n skips the samples of the first k of n optimizer
        # steps; the sampler order is seeded, so this continues the original run's data order.
        sampler = self._get_train_sampler_base(dataset)
        offset = _p40_resume_step()
        if offset <= 0:
            return sampler
        full_steps = int(os.environ["P40_FULL_STEPS"])
        if len(sampler) % full_steps != 0:
            raise ValueError(f"sampler length {len(sampler)} is not a multiple of P40_FULL_STEPS={full_steps}")
        skip = len(sampler) // full_steps * offset
        logger.warning(f"[p40] skipping the first {skip}/{len(sampler)} sampler indices ({offset}/{full_steps} steps)")
        return _P40SkipSampler(sampler, skip)

    def _get_train_sampler_base(self, dataset=None):
        if not self.diva_enabled:
            return super()._get_train_sampler(dataset)
        if dataset is None:
            dataset = self.train_dataset
        return DIVARepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=(
                self.args.generation_batch_size // self.num_generations
            ),
            repeat_count=(
                self.num_iterations * self.args.steps_per_generation
            ),
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _image_counts_per_sequence(self, input_ids: torch.Tensor) -> list[int]:
        """Count contiguous image-token blocks in each sequence."""
        if self.image_token_id is None:
            return [0] * input_ids.size(0)

        image_mask = input_ids.eq(self.image_token_id)
        block_starts = image_mask.clone()
        if image_mask.size(1) > 1:
            block_starts[:, 1:] &= ~image_mask[:, :-1]
        return block_starts.sum(dim=1).tolist()

    def _split_visual_inputs_by_sequence(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Group flattened OneVision image inputs by their owning sequence."""
        image_counts = self._image_counts_per_sequence(input_ids)
        if sum(image_counts) != image_grid_thw.size(0):
            per_sequence_counts = ",".join(str(count) for count in image_counts)
            raise ValueError(
                "Cannot align images with GRPO sequences: "
                f"found {sum(image_counts)} image-token blocks but "
                f"{image_grid_thw.size(0)} image grids. "
                f"Per-sequence image blocks=[{per_sequence_counts}]. "
                "A generated visual control token is the likely cause; "
                f"configured suppress_tokens={self.suppressed_visual_token_ids}."
            )

        pixel_lengths = image_grid_thw.prod(dim=1).tolist()
        if sum(pixel_lengths) != pixel_values.size(0):
            raise ValueError(
                "Cannot align image grids with pixel values: "
                f"grid lengths sum to {sum(pixel_lengths)}, but pixel_values "
                f"contains {pixel_values.size(0)} rows."
            )

        visual_inputs = []
        grid_offset = 0
        pixel_offset = 0
        for image_count in image_counts:
            next_grid_offset = grid_offset + image_count
            sequence_pixel_length = sum(pixel_lengths[grid_offset:next_grid_offset])
            next_pixel_offset = pixel_offset + sequence_pixel_length
            visual_inputs.append(
                (
                    image_grid_thw[grid_offset:next_grid_offset],
                    pixel_values[pixel_offset:next_pixel_offset],
                )
            )
            grid_offset = next_grid_offset
            pixel_offset = next_pixel_offset
        return visual_inputs

    @staticmethod
    def _merge_visual_inputs(batch: dict[str, Any]) -> dict[str, Any]:
        visual_inputs = batch.pop("_visual_inputs", None)
        if visual_inputs is None:
            return batch

        grids = [grid for grid, _ in visual_inputs if grid.numel() > 0]
        pixels = [pixel for _, pixel in visual_inputs if pixel.numel() > 0]
        if grids:
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
            batch["pixel_values"] = torch.cat(pixels, dim=0)
        return batch

    def _get_vgpo_backbone(self):
        """Return the multimodal backbone while keeping active PEFT adapters."""
        model = self.accelerator.unwrap_model(self.model)
        if is_peft_model(model):
            model = model.get_base_model()
        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RuntimeError(
                "VGPO needs a VLM exposing its multimodal backbone as "
                "`model.model`."
            )
        return backbone

    def _compute_vgpo_visual_focus(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run the VLM backbone once and extract detached visual-focus scores."""
        image_token_id = self.image_token_id
        if image_token_id is None:
            model = self.accelerator.unwrap_model(self.model)
            image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError(
                "VGPO could not find image_token_id on the processor or model."
            )

        visual_inputs = self._split_visual_inputs_by_sequence(
            input_ids, pixel_values, image_grid_thw
        )
        backbone = self._get_vgpo_backbone()
        batch_size = self.args.per_device_train_batch_size
        all_focus_scores = []

        with torch.inference_mode():
            for start in range(0, input_ids.size(0), batch_size):
                end = min(start + batch_size, input_ids.size(0))
                chunk_visual_inputs = visual_inputs[start:end]
                chunk_grids = [
                    grid for grid, _ in chunk_visual_inputs if grid.numel() > 0
                ]
                chunk_pixels = [
                    pixel for _, pixel in chunk_visual_inputs if pixel.numel() > 0
                ]
                if not chunk_grids:
                    raise RuntimeError(
                        "VGPO received a rollout chunk with no visual inputs."
                    )

                model_inputs = {
                    "input_ids": input_ids[start:end],
                    "attention_mask": attention_mask[start:end],
                    "image_grid_thw": torch.cat(chunk_grids, dim=0),
                    "pixel_values": torch.cat(chunk_pixels, dim=0),
                    "use_cache": False,
                }
                outputs = backbone(**model_inputs)
                last_hidden_state = getattr(
                    outputs, "last_hidden_state", None
                )
                if last_hidden_state is None:
                    try:
                        last_hidden_state = outputs[0]
                    except (KeyError, TypeError, IndexError) as exc:
                        raise RuntimeError(
                            "VGPO could not read the VLM's final hidden state."
                        ) from exc

                all_focus_scores.append(
                    compute_visual_focus_scores(
                        last_hidden_state=last_hidden_state,
                        input_ids=input_ids[start:end],
                        attention_mask=attention_mask[start:end],
                        completion_mask=completion_mask[start:end],
                        image_token_id=image_token_id,
                    )
                )

        return torch.cat(all_focus_scores, dim=0)

    def _attach_vgpo_weights(
        self, output: dict[str, Union[torch.Tensor, Any]]
    ) -> None:
        """Compute weights before TRL splits complete rollout groups."""
        required = (
            "prompt_ids",
            "prompt_mask",
            "completion_ids",
            "completion_mask",
            "pixel_values",
            "image_grid_thw",
        )
        missing = [key for key in required if key not in output]
        if missing:
            raise RuntimeError(
                "VGPO requires multimodal rollout tensors; missing "
                + ", ".join(missing)
            )

        input_ids = torch.cat(
            [output["prompt_ids"], output["completion_ids"]], dim=1
        )
        attention_mask = torch.cat(
            [output["prompt_mask"], output["completion_mask"]], dim=1
        )
        completion_mask = output["completion_mask"]
        focus_scores = self._compute_vgpo_visual_focus(
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            pixel_values=output["pixel_values"],
            image_grid_thw=output["image_grid_thw"],
        )

        use_intra = self.vgpo_config.get(
            "use_intra_trajectory_reweighting", True
        )
        normalized_focus, token_weights = compute_intra_trajectory_weights(
            focus_scores,
            completion_mask,
            use_visual_compensation=(
                use_intra
                and self.vgpo_config.get("use_visual_compensation", True)
            ),
            use_gated_visual_compensation=self.vgpo_config.get(
                "use_gated_visual_compensation", True
            ),
            visual_compensation_strength=self.vgpo_config.get(
                "visual_compensation_strength", 0.3
            ),
            gated_visual_compensation_start_ratio=self.vgpo_config.get(
                "gated_visual_compensation_start_ratio", 0.5
            ),
            visual_attention_threshold=self.vgpo_config.get(
                "visual_attention_threshold", 0.2
            ),
        )
        if not use_intra:
            token_weights = torch.ones_like(normalized_focus)

        if self.vgpo_config.get(
            "use_inter_trajectory_reweighting", True
        ):
            local_accumulation = compute_visual_accumulation(
                normalized_focus, completion_mask
            )
            global_accumulation = gather(local_accumulation)
            global_trajectory_weights = compute_inter_trajectory_weights(
                global_accumulation,
                self.num_generations,
            )
            local_batch_size = local_accumulation.numel()
            process_start = (
                self.accelerator.process_index * local_batch_size
            )
            trajectory_weights = global_trajectory_weights[
                process_start : process_start + local_batch_size
            ]
        else:
            trajectory_weights = torch.ones(
                completion_mask.size(0),
                device=completion_mask.device,
                dtype=torch.float32,
            )

        output["vgpo_token_weights"] = token_weights
        output["vgpo_trajectory_weights"] = trajectory_weights

        mode = "train" if self.model.training else "eval"
        valid = completion_mask.bool()
        valid_focus = normalized_focus[valid]
        valid_token_weights = token_weights[valid]
        if valid_focus.numel() > 0:
            self._metrics[mode]["vgpo/visual_focus_mean"].append(
                self.accelerator.gather(valid_focus.mean()).mean().item()
            )
            self._metrics[mode]["vgpo/token_weight_mean"].append(
                self.accelerator.gather(
                    valid_token_weights.mean()
                ).mean().item()
            )
        self._metrics[mode]["vgpo/trajectory_weight_mean"].append(
            self.accelerator.gather(
                trajectory_weights.mean()
            ).mean().item()
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # [p40] With gradient checkpointing on, HF models in train mode force use_cache=False inside
        # forward while generate() still feeds one token per step, which breaks Qwen3-VL's rope index.
        # Generation/scoring runs under no_grad and the model has no dropout, so eval mode is exact.
        _p40_eval = os.environ.get("P40_GRPO_SETUP") == "1" and self.model.training
        if _p40_eval:
            self.model.eval()
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            if _p40_eval:
                self.model.train()
        if self.diva_enabled:
            self._attach_diva_advantages(output)
        if self.vgpo_enabled:
            self._attach_vgpo_weights(output)
        return output

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Preserve multi-image sample boundaries through TRL's GRPO buffer."""
        mode = "train" if self.model.training else "eval"
        if mode != "train":
            return self._generate_and_score_completions(generation_batch)

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            generated = self._generate_and_score_completions(generation_batch)

            if "image_grid_thw" in generated and "pixel_values" in generated:
                generated["_visual_inputs"] = self._split_visual_inputs_by_sequence(
                    generated["prompt_ids"],
                    generated.pop("pixel_values"),
                    generated.pop("image_grid_thw"),
                )

            generated = shuffle_sequence_dict(generated)
            generation_batches = split_tensor_dict(
                generated, self.args.steps_per_generation
            )
            self._buffered_inputs = [
                self._merge_visual_inputs(batch) for batch in generation_batches
            ]

        inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        self._step += 1
        return inputs

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """TRL's log-probability pass with multi-image-aware slicing."""
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        visual_inputs = None
        if image_grid_thw is not None and pixel_values is not None:
            visual_inputs = self._split_visual_inputs_by_sequence(
                input_ids, pixel_values, image_grid_thw
            )

        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }

            if visual_inputs is not None:
                chunk_visual_inputs = visual_inputs[start:end]
                chunk_grids = [
                    grid for grid, _ in chunk_visual_inputs if grid.numel() > 0
                ]
                chunk_pixels = [
                    pixel for _, pixel in chunk_visual_inputs if pixel.numel() > 0
                ]
                if chunk_grids:
                    model_inputs["image_grid_thw"] = torch.cat(chunk_grids, dim=0)
                    model_inputs["pixel_values"] = torch.cat(chunk_pixels, dim=0)
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]

            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start:end]
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            all_logps.append(selective_log_softmax(logits, completion_ids))
            if compute_entropy:
                with torch.no_grad():
                    all_entropies.append(entropy_from_logits(logits))

        logps = torch.cat(all_logps, dim=0)
        entropies = (
            torch.cat(all_entropies, dim=0) if compute_entropy else None
        )
        return logps, entropies

    @profiling_decorator
    def _get_pepo_per_token_outputs(
        self,
        model,
        input_ids,
        attention_mask,
        completion_mask,
        logits_to_keep,
        batch_size=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return policy log-probs, entropy, and PEPO token weights."""
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        all_token_weights = []

        visual_inputs = None
        if image_grid_thw is not None and pixel_values is not None:
            visual_inputs = self._split_visual_inputs_by_sequence(
                input_ids, pixel_values, image_grid_thw
            )

        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "return_dict": True,
                "output_hidden_states": True,
            }

            if visual_inputs is not None:
                chunk_visual_inputs = visual_inputs[start:end]
                chunk_grids = [
                    grid for grid, _ in chunk_visual_inputs if grid.numel() > 0
                ]
                chunk_pixels = [
                    pixel
                    for _, pixel in chunk_visual_inputs
                    if pixel.numel() > 0
                ]
                if chunk_grids:
                    model_inputs["image_grid_thw"] = torch.cat(
                        chunk_grids, dim=0
                    )
                    model_inputs["pixel_values"] = torch.cat(
                        chunk_pixels, dim=0
                    )
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]

            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[
                    start:end
                ]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start:end]
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            outputs = model(**model_inputs)
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :] / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            entropies = entropy_from_logits(logits)

            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError(
                    "PEPO requested output_hidden_states=True, but the model "
                    "did not return hidden states."
                )
            vision_scores = compute_vision_scores(
                hidden_states,
                input_ids_batch,
                logits_to_keep,
                self.pepo_image_token_id,
                self.pepo_config["vision_score_type"],
            )
            token_weights = compute_pepo_token_weights(
                vision_scores,
                entropies,
                completion_mask[start:end].bool(),
                gate_alpha=self.pepo_config["gate_alpha"],
                gate_temperature=self.pepo_config["gate_temperature"],
            )

            all_logps.append(logps)
            all_entropies.append(entropies)
            all_token_weights.append(token_weights)

        return (
            torch.cat(all_logps, dim=0),
            torch.cat(all_entropies, dim=0),
            torch.cat(all_token_weights, dim=0),
        )

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        try:
            golds = reward_kwargs.pop("ground_truth")
        except KeyError as exc:
            raise ValueError(
                "GRPO samples must provide a separate 'ground_truth' field"
            ) from exc

        # The model may receive multiple PIL images, while the visual reward
        # functions compare the original positive and negative image paths.
        reward_images = reward_kwargs.pop("reward_image", None)
        if reward_images is not None:
            reward_kwargs["image"] = reward_images

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, golds=golds, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        if self.diva_enabled:
            metadata_keys = (
                "diva_source_id",
                "diva_source_index",
                "diva_variant_id",
                "diva_base_difficulty",
                "diva_difficulty",
            )
            missing = [
                key for key in metadata_keys if key not in inputs[0]
            ]
            if missing:
                raise RuntimeError(
                    "DIVA rollout metadata was not preserved: "
                    + ", ".join(missing)
                )
            self._diva_last_rewards = rewards_per_func.detach()
            self._diva_last_metadata = {
                key: gather_object([example[key] for example in inputs])
                for key in metadata_keys
            }
        return rewards_per_func

    def _attach_diva_advantages(
        self, output: dict[str, Union[torch.Tensor, Any]]
    ) -> None:
        if self._diva_last_rewards is None or self._diva_last_metadata is None:
            raise RuntimeError(
                "DIVA reward metadata is unavailable for advantage computation."
            )

        metadata = self._diva_last_metadata
        device = self.accelerator.device
        rewards = (
            self._diva_last_rewards
            * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        all_advantages, diagnostics = compute_diva_advantages(
            rewards,
            [str(value) for value in metadata["diva_variant_id"]],
            [str(value) for value in metadata["diva_source_id"]],
            metadata["diva_difficulty"],
            weight_mode=self.diva_config["weight_mode"],
            weighted_advantage_k=self.diva_config[
                "weighted_advantage_k"
            ],
            rrb_threshold=self.diva_config["rrb_threshold"],
            rrb_scale_factor=self.diva_config["rrb_scale_factor"],
            local_weight=self.diva_config["local_weight"],
            global_weight=self.diva_config["global_weight"],
        )

        local_batch_size = output["advantages"].numel()
        process_start = self.accelerator.process_index * local_batch_size
        process_end = process_start + local_batch_size
        if process_end > all_advantages.numel():
            raise RuntimeError(
                "DIVA could not align gathered advantages with the local batch."
            )
        output["advantages"] = all_advantages[process_start:process_end]

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["diva/local_advantage_mean"].append(
            diagnostics["local"].mean().item()
        )
        self._metrics[mode]["diva/global_advantage_mean"].append(
            diagnostics["global"].mean().item()
        )
        self._metrics[mode]["diva/combined_advantage_abs_mean"].append(
            all_advantages.abs().mean().item()
        )
        self._metrics[mode]["diva/difficulty_mean"].append(
            torch.as_tensor(
                metadata["diva_difficulty"], dtype=torch.float32
            ).mean().item()
        )

        advantage_log = self._logs["advantages"]
        for _ in range(min(len(advantage_log), all_advantages.numel())):
            advantage_log.pop()
        advantage_log.extend(all_advantages.tolist())

        if mode == "train" and hasattr(
            self.train_dataset, "update_difficulties"
        ):
            updates = compute_difficulty_updates(
                rewards,
                metadata["diva_source_index"],
                metadata["diva_base_difficulty"],
                self.diva_config["score_ranges"],
                self.diva_config["difficulty_changes"],
                self.diva_config["min_difficulty"],
                self.diva_config["max_difficulty"],
            )
            self.train_dataset.update_difficulties(updates)

    def _compute_loss(self, model, inputs):
        if not self.vgpo_enabled and not self.pepo_enabled:
            return super()._compute_loss(model, inputs)

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        pepo_token_weights = None
        if self.pepo_enabled and self.pepo_use_vision_weights:
            (
                per_token_logps,
                entropies,
                pepo_token_weights,
            ) = self._get_pepo_per_token_outputs(
                model,
                input_ids,
                attention_mask,
                completion_mask,
                logits_to_keep,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
            )
        else:
            (
                per_token_logps,
                entropies,
            ) = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
            )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = get_high_entropy_mask(
                entropies,
                completion_mask,
                1 - self.top_entropy_quantile,
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        if self.vgpo_enabled:
            token_weights = inputs.get("vgpo_token_weights")
            trajectory_weights = inputs.get("vgpo_trajectory_weights")
            if token_weights is None or trajectory_weights is None:
                raise RuntimeError(
                    "VGPO rollout weights were not preserved in the TRL "
                    "buffer."
                )
            advantages = (
                inputs["advantages"].unsqueeze(1)
                * token_weights.to(per_token_logps.dtype)
                * trajectory_weights.to(
                    per_token_logps.dtype
                ).unsqueeze(1)
            )
        else:
            advantages = inputs["advantages"].unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (
                (log_ratio * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)
        else:
            raise ValueError(
                "Unknown importance sampling level: "
                f"{self.importance_sampling_level}. Possible values are "
                "'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(
            coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high
        )
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        annealed_pepo_weights = None
        if pepo_token_weights is not None:
            annealed_pepo_weights = anneal_pepo_token_weights(
                pepo_token_weights,
                global_step=self.state.global_step,
                max_steps=self.state.max_steps,
            )
            per_token_loss = per_token_loss * annealed_pepo_weights
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                (per_token_loss * completion_mask).sum()
                / completion_mask.sum().clamp(min=1.0)
            )
        elif self.loss_type == "dr_grpo":
            loss = (
                (per_token_loss * completion_mask).sum()
                / (per_token_loss.size(0) * self.max_completion_length)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(values):
            if values.shape[1] == 1:
                return values.mean()
            return (
                (values * completion_mask).sum() / completion_token_count
            )

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )
        if self.vgpo_enabled:
            mean_weighted_advantage = masked_batch_mean(advantages)
            self._metrics[mode]["vgpo/weighted_advantage_mean"].append(
                self.accelerator.gather(
                    mean_weighted_advantage
                ).nanmean().item()
            )
        if pepo_token_weights is not None:
            raw_weight_mean = masked_batch_mean(pepo_token_weights)
            annealed_weight_mean = masked_batch_mean(
                annealed_pepo_weights
            )
            self._metrics[mode]["pepo/token_weight_mean"].append(
                self.accelerator.gather(
                    raw_weight_mean
                ).nanmean().item()
            )
            self._metrics[mode]["pepo/annealed_weight_mean"].append(
                self.accelerator.gather(
                    annealed_weight_mean
                ).nanmean().item()
            )

        is_low_clipped = (
            (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        )
        is_high_clipped = (
            (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )
        return loss
