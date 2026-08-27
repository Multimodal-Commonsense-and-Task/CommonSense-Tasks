"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

This file is the SCALE-enabled drop-in replacement for OpenVLA's `modeling_prismatic.py`. On top of the original
OpenVLA model it implements `predict_action` with the following inference-time decoding strategies:

    - "greedy" : standard greedy (argmax) decoding (OpenVLA default)
    - "temp"   : temperature sampling
    - "topk"   : top-k sampling (with temperature)
    - "topp"   : top-p (nucleus) sampling
    - "scale"  : SCALE -- Self-uncertainty Conditioned Adaptive Looking and Execution

SCALE (Sec. 3 of the paper) jointly modulates:
    (1) action decoding   -- per-token sampling temperature  tau_k = T0 * sigmoid(u_k)            (Eq. 4)
    (2) visual perception -- vision-encoder attention temperature  gamma = kappa^tanh(s * d_u)     (Eq. 8)
where the self-uncertainty u_k = D_KL(p || q_low) - D_KL(p || q_high) is the expected log-likelihood ratio
between a one-hot (full-certainty) and a uniform (full-ambiguity) reference distribution (Eq. 2-3).

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100

# === OpenVLA action-token range (the last 256 entries of the LLM vocabulary are the action bins) ===
ACTION_TOKEN_BEGIN = 31744  # logits[:ACTION_TOKEN_BEGIN] are masked out (non-action tokens)
ACTION_TOKEN_END = 32000    # logits[ACTION_TOKEN_END:] are masked out (padding tokens)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === Adaptive Visual Attention (SCALE, Sec. 3.3.2) ===
def _create_attention_forward_with_custom_scale(attn_module):
    """Create a custom forward function that respects modified attention `scale` values.

    TIMM's fused attention path ignores `self.scale`, so we fall back to an explicit (manual) attention
    implementation that honors the temperature-scaled `scale`.
    """
    def custom_forward(x):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)

        # Always use manual attention to respect custom scale
        # (fused attention ignores self.scale modifications)
        q = q * attn_module.scale  # Use the potentially modified scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    return custom_forward


def apply_visual_attention_temperature(vision_backbone: nn.Module, temperature: float) -> None:
    """
    Apply temperature scaling (gamma) to every self-attention layer of the vision encoder (SCALE, Eq. 9).

    Args:
        vision_backbone: The PrismaticVisionBackbone module
        temperature: gamma -- the attention temperature to apply (multiplies the attention scale)

    Note:
        - TIMM's attention uses scale = head_dim^-0.5
        - To apply temperature gamma, we modify: scale_new = scale_original / gamma
        - Higher gamma -> lower scale -> softer (flatter) attention  -> broader exploration
        - Lower  gamma -> higher scale -> sharper attention          -> focused perception
        - This function also monkey-patches the attention forward to ensure the scale is used
          even when fused_attn is enabled.
        - For OpenVLA's fused DINOv2 + SigLIP backbone, gamma is applied to BOTH encoders (Appendix G.3.2).
    """
    if not hasattr(vision_backbone, 'featurizer'):
        return

    def _patch(featurizer):
        if not hasattr(featurizer, 'blocks'):
            return
        for block in featurizer.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'scale'):
                # Recompute the original scale (head_dim^-0.5) each time to ensure consistency
                if hasattr(block.attn, 'head_dim'):
                    original_scale = block.attn.head_dim ** -0.5
                else:
                    original_scale = block.attn.scale

                # Apply temperature: scale_new = scale_original / gamma
                block.attn.scale = original_scale / max(temperature, 1e-8)

                # Monkey-patch the forward method to ensure scale is used (fused attention ignores it)
                if not hasattr(block.attn, '_original_forward'):
                    block.attn._original_forward = block.attn.forward
                block.attn.forward = _create_attention_forward_with_custom_scale(block.attn)

    # Apply to main featurizer (and to the fused featurizer for OpenVLA's DINOv2 + SigLIP backbone)
    _patch(vision_backbone.featurizer)
    if getattr(vision_backbone, 'use_fused_vision_backbone', False) and hasattr(vision_backbone, 'fused_featurizer'):
        _patch(vision_backbone.fused_featurizer)


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    # === Self-Uncertainty (SCALE, Sec. 3.2) ===
    def _compute_self_uncertainty(self, step_logits, num_logits, epsilon):
        """Compute the token-level self-uncertainty u_k for a single decoding step (Eq. 2-3).

            u_k = D_KL(p || q_low) - D_KL(p || q_high) = E_{x~p}[ log( q_high(x) / q_low(x) ) ]

        where q_low is a one-hot (full-certainty) reference on the top-1 token and q_high is uniform
        (full-ambiguity). Both references and the predicted distribution p are taken over the top-`num_logits`
        action tokens (num_logits = action-vocab size recovers the exact paper formulation over all of |V|).

        Returns: (u_k as a Python float, top_indices, probs over the masked action vocabulary).
        """
        # Mask out every non-action logit (keep only the 256 action-bin tokens)
        masked_logits = step_logits.clone()
        masked_logits[:ACTION_TOKEN_BEGIN] = float('-inf')
        masked_logits[ACTION_TOKEN_END:] = float('-inf')
        probs = torch.softmax(masked_logits, dim=-1)

        # Work with the top-K action tokens (K = min(num_logits, 256); num_logits <= 0 => all 256)
        K = min(num_logits, 256) if num_logits > 0 else 256
        top_probs, top_indices = torch.topk(probs, k=K, largest=True, sorted=True)

        # Renormalize the (truncated) predicted distribution p
        p = top_probs / top_probs.sum()

        # Low-uncertainty reference q_low: one-hot on the top-1 token, smoothed by epsilon
        q_low = torch.full_like(p, epsilon)
        q_low[0] = 1.0 - (K - 1) * epsilon

        # High-uncertainty reference q_high: uniform over the K tokens
        q_high = torch.ones_like(p) / K

        # u_k = E_{x~p}[ log( q_high(x) / q_low(x) ) ]   (Eq. 3)
        u_k = torch.sum(p * torch.log(q_high / q_low)).item()
        return u_k, top_indices, probs

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        decoding_mode: str = "greedy",
        # --- Baseline sampling hyperparameters ---
        temperature: float = 1.0,        # "temp"/"topk": sampling temperature T
        top_k: int = 40,                 # "topk": number of top tokens to keep (k)
        top_p: float = 0.9,              # "topp": nucleus probability mass (p)
        # --- SCALE hyperparameters (paper Sec. 3; defaults live in configs/scale.yaml) ---
        T0: float = 1.0,                 # base/max sampling temperature; tau_k = T0 * sigmoid(u_k)  (Eq. 4)
        kappa: float = 2.0,              # visual-attention bound; gamma in (1/kappa, kappa)         (Eq. 8)
        alpha: float = 0.8,              # EMA temporal smoothing of step-level uncertainty          (Eq. 7)
        epsilon: float = 1e-12,          # smoothing for the one-hot low-uncertainty reference q_low
        num_logits: int = 256,           # # of top action logits used to compute u (256 = full action vocab)
        attn_sensitivity: float = 0.3,   # sensitivity multiplier inside tanh(sensitivity * delta_u)
        uncertainty_history: Optional[List[List[float]]] = None,  # per-dim [ema_u, last_u] threaded across the episode
        **kwargs: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict a single action (a sequence of `action_dim` discrete tokens) from the current observation.

        Args:
            input_ids: Tokenized prompt (image placeholder + instruction).
            unnorm_key: Key into `norm_stats` used to un-normalize the predicted action.
            decoding_mode: One of {"greedy", "temp", "topk", "topp", "scale"}.
            temperature: Sampling temperature for "temp" and "topk".
            top_k: Number of highest-probability tokens to keep for "topk".
            top_p: Nucleus probability mass for "topp".
            T0, kappa, alpha, epsilon, num_logits, attn_sensitivity: SCALE hyperparameters (see configs/scale.yaml).
            uncertainty_history: SCALE state threaded across the closed-loop episode -- a list
                [ema_u_per_dim, last_u_per_dim]. Pass `None` on the first step; pass back the value
                returned in the info dict on subsequent steps.
            **kwargs: Extra generation arguments (e.g. `attention_mask`, `pixel_values`).

        Returns:
            (actions, info) where `actions` is a (action_dim,) np.ndarray of un-normalized continuous actions
            and `info` carries the updated `uncertainty_history` (for "scale"; `None` otherwise).
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        action_dim = self.get_action_dim(unnorm_key)

        # ============================================================================================
        # SCALE -- Adaptive Looking and Execution (Algorithm 1)
        # ============================================================================================
        if decoding_mode == "scale":
            # ---- (Sec. 3.3.2) Adaptive Visual Attention: gamma_t = kappa^tanh(s * delta_u_{t-1}) ----
            # delta_u_{t-1} = u_{t-1} - EMA_{t-2} is the change in step-level uncertainty from recent history.
            # On the first step (no history yet) we apply gamma = 1.0 (no modulation).
            if uncertainty_history is None:
                uncertainty_history = [[0.0] * action_dim, [0.0] * action_dim]
            ema_u, last_u = uncertainty_history[0], uncertainty_history[1]

            if all(u == 0.0 for u in ema_u):
                gamma = 1.0
            else:
                mean_delta_u = np.mean([last_u[i] - ema_u[i] for i in range(action_dim)])
                gamma = kappa ** np.tanh(attn_sensitivity * mean_delta_u)
            apply_visual_attention_temperature(self.vision_backbone, gamma)

            # ---- (Sec. 3.3.1) Adaptive Action Decoding with a single forward pass + KV cache ----
            predicted_action_token_ids = []
            new_u = []

            past_key_values = None
            current_input_ids = input_ids
            attention_mask = kwargs.get("attention_mask", None)
            pixel_values = kwargs.get("pixel_values", None)

            for step_idx in range(action_dim):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self(
                            input_ids=current_input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        outputs = self(
                            input_ids=current_input_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )

                step_logits = outputs.logits[:, -1, :][0]   # (vocab_size,)
                past_key_values = outputs.past_key_values

                # Self-uncertainty u_k for this token (Eq. 2-3)
                u_k, top_indices, _ = self._compute_self_uncertainty(step_logits, num_logits, epsilon)
                new_u.append(u_k)

                # Per-token sampling temperature tau_k = T0 * sigmoid(u_k)  (Eq. 4)
                tau_k = T0 * torch.sigmoid(torch.tensor(u_k)).item()

                if tau_k < 1e-3:
                    # Near-greedy execution under low uncertainty (tau ~ 0)
                    selected_token = top_indices[0].item()
                else:
                    # Explorative sampling: a_k ~ Cat(softmax(l_k / tau_k))  (Eq. 5)
                    top_logits = step_logits[top_indices]
                    scaled_probs = torch.softmax(top_logits / tau_k, dim=-1)
                    sampled_idx = torch.multinomial(scaled_probs, num_samples=1).item()
                    selected_token = top_indices[sampled_idx].item()

                predicted_action_token_ids.append(selected_token)
                current_input_ids = torch.tensor([[selected_token]], device=input_ids.device)

            # ---- Update step-level uncertainty history (EMA, Eq. 6-7) ----
            if all(u == 0.0 for u in ema_u):
                # First step: initialize EMA = current (no smoothing yet)
                uncertainty_history[0] = new_u.copy()
                uncertainty_history[1] = new_u.copy()
            else:
                # EMA_t = alpha * EMA_{t-1} + (1 - alpha) * u_{t-1}; last_u <- u_t
                uncertainty_history[0] = [alpha * ema_u[i] + (1.0 - alpha) * last_u[i] for i in range(action_dim)]
                uncertainty_history[1] = new_u

            predicted_action_token_ids = np.array(predicted_action_token_ids)
            info = {"uncertainty_history": uncertainty_history}

        # ============================================================================================
        # Baseline decoding strategies (greedy / temperature / top-k / top-p)
        # ============================================================================================
        else:
            outputs = self.generate(
                input_ids,
                max_new_tokens=action_dim,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
            generated_ids = outputs.sequences
            scores = outputs.scores  # tuple of (batch, vocab) logit tensors, one per decoded token

            if decoding_mode == "temp":
                # Temperature sampling over the action vocabulary (paper baseline: "Sampling (t)")
                predicted_action_token_ids = []
                for step_logits in scores:
                    if temperature == 0.0:
                        selected_token = torch.argmax(step_logits[0]).item()
                    else:
                        sampling_logits = step_logits[0] / temperature
                        sampling_logits[:ACTION_TOKEN_BEGIN] = -float('inf')
                        sampling_logits[ACTION_TOKEN_END:] = -float('inf')
                        sampling_probs = torch.softmax(sampling_logits, dim=-1)
                        selected_token = torch.multinomial(sampling_probs, num_samples=1).item()
                    predicted_action_token_ids.append(selected_token)
                predicted_action_token_ids = np.array(predicted_action_token_ids)

            elif decoding_mode == "topk":
                # Top-k sampling with temperature (paper baseline: "Top-k (k, t)")
                predicted_action_token_ids = []
                for step_logits in scores:
                    sampling_logits = step_logits[0] / temperature
                    sampling_logits[:ACTION_TOKEN_BEGIN] = -float('inf')
                    sampling_logits[ACTION_TOKEN_END:] = -float('inf')

                    # Restrict to the top-k tokens within the valid action range
                    valid_logits = sampling_logits[ACTION_TOKEN_BEGIN:ACTION_TOKEN_END]
                    k = min(top_k, len(valid_logits))
                    _, top_k_in_valid = torch.topk(valid_logits, k=k)
                    mask = torch.ones(len(valid_logits), dtype=torch.bool, device=valid_logits.device)
                    mask[top_k_in_valid] = False
                    sampling_logits[ACTION_TOKEN_BEGIN:ACTION_TOKEN_END][mask] = -float('inf')

                    sampling_probs = torch.softmax(sampling_logits, dim=-1)
                    selected_token = torch.multinomial(sampling_probs, num_samples=1).item()
                    predicted_action_token_ids.append(selected_token)
                predicted_action_token_ids = np.array(predicted_action_token_ids)

            elif decoding_mode == "topp":
                # Top-p (nucleus) sampling over the action vocabulary (paper baseline: "Top-p (p)")
                predicted_action_token_ids = []
                for step_logits in scores:
                    sampling_logits = step_logits[0].clone()
                    sampling_logits[:ACTION_TOKEN_BEGIN] = -float('inf')
                    sampling_logits[ACTION_TOKEN_END:] = -float('inf')
                    sampling_probs = torch.softmax(sampling_logits, dim=-1)

                    # Sort by descending probability, keep the smallest nucleus with cumulative mass >= top_p
                    sorted_probs, sorted_indices = torch.sort(sampling_probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs >= top_p
                    # Shift right so the first token crossing the threshold is kept (always keep top-1)
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    final_logits = sampling_logits.clone()
                    final_logits[sorted_indices[sorted_indices_to_remove]] = -float('inf')
                    final_probs = torch.softmax(final_logits, dim=-1)
                    selected_token = torch.multinomial(final_probs, num_samples=1).item()
                    predicted_action_token_ids.append(selected_token)
                predicted_action_token_ids = np.array(predicted_action_token_ids)

            else:  # decoding_mode == "greedy"
                # Standard greedy (argmax) decoding -- the OpenVLA default
                predicted_action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()

            info = {"uncertainty_history": None}

        # ============================================================================================
        # De-tokenize action-token ids -> normalized actions -> un-normalized continuous actions
        # ============================================================================================
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, info

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
