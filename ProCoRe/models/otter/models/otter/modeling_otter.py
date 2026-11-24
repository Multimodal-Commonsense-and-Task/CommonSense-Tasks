import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from einops import rearrange, repeat
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoTokenizer
from ..falcon.modelling_RW import RWForCausalLM
from ..mpt.modeling_mpt import MPTForCausalLM
from ..mpt_redpajama.mosaic_gpt import MosaicGPT
from .configuration_otter import OtterConfig

from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaAttention, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.nn import CrossEntropyLoss
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack

import math
import copy
from transformers.utils import logging
from functools import partial


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

import torch.distributed as dist


def master_print(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


# Add this line at the beginning of your script or in your main function
# dist.init_process_group(backend='nccl')

XFORMERS_AVAIL = False
XFORMERS_MSG_PRINTED = False  # Add this global variable
try:
    if not XFORMERS_MSG_PRINTED:  # Check if the message has been master_printed before
        import xformers.ops as xops
        from transformers import LlamaTokenizer

        from xformers_model import CLIPVisionModel, LlamaForCausalLM

        _xformers_version = importlib_metadata.version("xformers")
        if dist.is_initialized() and dist.get_rank() == 0:  # Check if the current process rank is 0
            master_print(f"Successfully imported xformers version {_xformers_version}")
except ImportError as e:
    if not XFORMERS_MSG_PRINTED:  # Check if the message has been master_printed before
        from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

        if dist.is_initialized() and dist.get_rank() == 0:  # Check if the current process rank is 0
            master_print(f"Failed to import xformers: {e}")
            XFORMERS_AVAIL = False
            master_print("No xformers found. You are recommended to install xformers via `pip install xformers` or `conda install -c xformers xformers`")
            XFORMERS_MSG_PRINTED = True  # Set the variable to True after master_printing the message

# from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "RWForCausalLM": "transformer.h",
    "MPTForCausalLM": "transformer.blocks",
    "MosaicGPT": "transformer.blocks",
}

MODEL_CLASSES = {
    "LlamaForCausalLM": "llama",
    "OPTForCausalLM": "opt",
    "GPTJForCausalLM": "gptj",
    "GPTNeoXForCausalLM": "gpt_neox",
    "MPTForCausalLM": "mpt",
    "MosaicGPT": "mpt",
}


def _infer_decoder_layers_attr_name(model: nn.Module):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually.")


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls), {})  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def exists(val):
    return val is not None



def consecutive_ranges(t: torch.Tensor, target_val: int):
    mask = (t == target_val)

    padded = torch.cat([torch.tensor([False], device=t.device),
                        mask,
                        torch.tensor([False], device=t.device)])

    diff = padded[1:] ^ padded[:-1]  # True where state changes

    change_idx = diff.nonzero(as_tuple=False).flatten()

    starts = change_idx[0::2]
    ends   = change_idx[1::2] - 1
    
    starts, ends = starts.tolist(), ends.tolist()
    
    starts = [start+(ind*224) for ind,start in enumerate(starts)]
    ends = [start+224-1 for start in starts]

    return list(zip(starts, ends))


logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    dara_amplifier = None,
    use_dara = False,
    **kwargs,
):
    img_pos_ranges = kwargs["img_pos_ranges"]
    
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    
    # DARA (logits column scaling) — only if explicitly enabled
    if use_dara:
        triples = []
        if isinstance(img_pos_ranges, (list, tuple)) and len(img_pos_ranges) > 0:
            next_slot = 0
            for rng in img_pos_ranges:
                if len(rng) == 2:
                    s, e = int(rng[0]), int(rng[1])
                    triples.append((0, s, e, next_slot))  # auto slot 0,1,2,...
                    next_slot += 1
                elif len(rng) == 3:
                    s, e, sl = int(rng[0]), int(rng[1]), int(rng[2])
                    triples.append((0, s, e, sl))
                else:
                    raise ValueError("img_pos_ranges element must be (s,e) or (s,e,slot)")
            img_pos_tensor = torch.as_tensor(triples, device=attn_weights.device, dtype=torch.long)
            finite_mask = torch.isfinite(attn_weights)  # defensive
            attn_weights = dara_amplifier(attn_weights, img_pos_tensor)
            attn_weights = torch.where(finite_mask, attn_weights, attn_weights.new_full((), float("-inf")))
    # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #     raise ValueError(
    #         f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
    #         f' {attn_weights.size()}'
    #     )
   
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class AmplifiedAttentionHeadFactor(nn.Module):
    def __init__(self, attention_amplify_head_ind, max_images: int = 0):
        """
        attention_amplify_head_ind: iterable of head indices to amplify.
        max_images: number of columns for image slots (amplify_factor's 2nd dim).
        """
        super().__init__()
        # Store selected heads as a buffer (compile-friendly; not a Parameter)
        head_idx = list(map(int, attention_amplify_head_ind))
        self.register_buffer(
            "head_idx_tensor",
            torch.as_tensor(head_idx, dtype=torch.long),
            persistent=False,
        )
        # Learnable amplification matrix: [#selected_heads, max_images], init to 1.0
        self.amplify_factor = nn.Parameter(
            torch.ones(len(head_idx), max_images, dtype=torch.bfloat16),
            requires_grad=True,
        )
        self.max_images = max_images
        print("init amplifier")
    @torch.no_grad()
    def _interval_mask_single(self, B, K, b: int, s: int, e: int, device, dtype):
        """Return a [B,1,1,K] mask for a single inclusive interval [s, e] on batch b."""
        s = max(0, min(int(s), K - 1))
        e = max(0, min(int(e), K - 1))
        if e < s:
            s, e = e, s
        mask = torch.zeros((B, 1, 1, K), device=device, dtype=dtype)
        mask[b, 0, 0, s:e + 1] = 1
        return mask

    def forward(self, attention_scores: torch.Tensor, img_pos: torch.Tensor):
        """
        attention_scores: [B, H, Q, K] attention logits BEFORE mask & softmax.
        img_pos: LongTensor [N, 4], each row = (batch_idx, start, end, slot), inclusive [start, end].
                - slot selects the column in amplify_factor (modulo max_images if desired).
                - If multiple rows share same slot for the same batch, their masks add up (they are unioned).
        """
        B, H, Q, K = attention_scores.shape
        device = attention_scores.device
        dtype  = attention_scores.dtype

        if img_pos is None or img_pos.numel() == 0:
            return attention_scores

        img_pos = img_pos.to(device=device)  # <-- device only!
        # expected shape [N, 4]
        if img_pos.size(1) != 4:
            raise ValueError("img_pos must be of shape [N, 4] with (batch_idx, start, end, slot).")

        # Start from ones so columns outside any image keep scale 1.
        # We accumulate per-row contribution: combined = 1 + Σ_row ( delta(head, slot) * mask_row )
        combined = torch.ones((B, H, 1, K), device=device, dtype=dtype)

        # Prebuild a "ones per head" baseline and the selected-heads index once.
        heads_all_ones = torch.ones((H,), device=device, dtype=dtype)

        # Loop over rows (N = #image intervals across batch). Typically small.
        for i in range(img_pos.size(0)):
            b = int(img_pos[i, 0].item())
            s = int(img_pos[i, 1].item())
            e = int(img_pos[i, 2].item())
            raw_slot = int(img_pos[i, 3].item())

            # Map slot into [0, max_images-1]
            col = raw_slot % self.max_images

            # Build the interval mask for this row: [B,1,1,K]
            mask_row = self._interval_mask_single(B, K, b, s, e, device, dtype)

            # Build per-head scale for this row.
            # Default = 1 for all heads; selected heads use learnable factors at column `col`.
            scales_H = heads_all_ones.clone()
            if self.head_idx_tensor.numel() > 0:
                # NOTE: fix the earlier bug — use dtype=... not device=...
                learnable_col = self.amplify_factor[:, col].to(dtype=dtype)  # [#selected_heads]
                scales_H.scatter_(0, self.head_idx_tensor, learnable_col)
            # delta = (scale - 1) per head → [1,H,1,1] for broadcast.
            delta = (scales_H - 1.0).view(1, H, 1, 1)

            # Accumulate: 1 + delta * mask_row
            combined = combined + delta * mask_row  # broadcast over B,H,K

        # Multiply attention logits in-place (safe for non-leaf tensors)
        attention_scores.mul_(combined)  # [B,H,Q,K] *= [B,H,1,K]

        return attention_scores



class My_LlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, use_dara: bool=False, use_only: bool=False, max_images: int=0):
        super().__init__(config=config, layer_idx=layer_idx)
        self.num_heads = config.num_attention_heads
        self.max_images = max_images
        if use_dara:
            # Amplifier used by DARA (select all heads by default)
            self._amp_head_idx = list(range(self.num_heads))
            self.amplifier = AmplifiedAttentionHeadFactor(self._amp_head_idx, max_images=max_images)
        self.use_dara = use_dara
        self.use_only = use_only
        self.layer_idx = layer_idx

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager" and not self.use_dara:
            if self.config._attn_implementation == "sdpa" or kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                dara_amplifier=self.amplifier,
                use_dara=self.use_dara,
                use_only=self.use_only,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class My_LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, use_dara: bool=False, use_only: bool=False,  max_images: int=0):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = My_LlamaAttention(config=config, layer_idx=layer_idx, use_dara=use_dara, use_only=use_only, max_images=max_images)

class My_LlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, use_dara: bool=False, use_only: bool=False, max_images: int=0):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        if use_dara:
            self.layers = nn.ModuleList([
                My_LlamaDecoderLayer(
                    config,
                    layer_idx,
                    use_dara=use_dara if layer_idx == 0 else False,
                    max_images=max_images if layer_idx == 0 else 0,
                )
                for layer_idx in range(config.num_hidden_layers)
            ])
        elif use_only:
            self.layers = nn.ModuleList(
                [My_LlamaDecoderLayer(config, layer_idx, use_only=use_only) for layer_idx in range(config.num_hidden_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [My_LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        img_pos_ranges: Optional[object] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    img_pos_ranges=img_pos_ranges
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    img_pos_ranges=img_pos_ranges,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class OtterPerceiverBlock(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8, mult: int = 4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        ff_dim = dim * mult
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(ff_dim, dim, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        residual_latents = latents
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        out = self.to_out(out) + residual_latents
        residual_out = out
        for layer in self.feed_forward:
            out = layer(out)
        return out + residual_out


class OtterPerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        # max_num_frames: int = 128,
        max_num_media: Optional[int] = None,
        max_num_frames: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if exists(max_num_frames) else None

        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(OtterPerceiverBlock(dim=dim, dim_head=dim_head, heads=heads, mult=ff_mult))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for block in self.layers:
            latents = block(x, latents)
        return self.norm(latents)


class OtterMaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        only_attend_immediate_media: bool = True,
        use_dara: bool=False,
        use_only: bool=False,
        max_images: int=0
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: Optional[torch.BoolTensor] = None,
        attend_previous: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            attend_previous: bool
                If false, ignores immediately preceding image and starts attending when following image
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        if not XFORMERS_AVAIL:
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k = rearrange(k, "b n (h d) -> b h n d", h=h)
            v = rearrange(v, "b n (h d) -> b h n d", h=h)
            q = q * self.scale

            sim = torch.einsum("... i d, ... j d -> ... i j", q, k)
            if exists(media_locations):
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)
                media_time = torch.arange(T_img, device=x.device) + 1

                if not attend_previous:
                    text_time[~media_locations] += 1
                    # make sure max is still the number of images in the sequence
                    text_time[
                        text_time
                        > repeat(
                            torch.count_nonzero(media_locations, dim=1),
                            "b -> b i",
                            i=text_time.shape[1],
                        )
                    ] = 0

                # text time must equal media time if only attending to most immediate image
                # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
                mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

                text_to_media_mask = mask_op(
                    rearrange(text_time, "b i -> b 1 i 1"),
                    repeat(media_time, "j -> 1 1 1 (j n)", n=n),
                )
                sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

            sim = sim - sim.amax(dim=-1, keepdim=True).detach()
            attn = sim.softmax(dim=-1)

            if exists(media_locations) and self.only_attend_immediate_media:
                # any text without a preceding media needs to have attention zeroed out
                text_without_media_mask = text_time == 0
                text_without_media_mask = rearrange(text_without_media_mask, "b i -> b 1 i 1")
                attn = attn.masked_fill(text_without_media_mask, 0.0)

            out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            q = rearrange(q, "b n (h d) -> b n h d", h=h)
            k = rearrange(k, "b n (h d) -> b n h d", h=h)
            v = rearrange(v, "b n (h d) -> b n h d", h=h)
            attn_mask = None
            out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=self.scale)
        return self.to_out(out)


class OtterGatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
        use_dara: bool=False,
        use_only: bool=False,
        max_images: int=0
    ):
        super().__init__()
        self.attn = OtterMaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
            max_images=max_images,
            use_dara=use_dara,
            use_only=use_only,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * ff_mult, bias=False),
                nn.GELU(),
                nn.Linear(dim * ff_mult, dim, bias=False),
            ]
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: Optional[torch.BoolTensor] = None,
        attend_previous: bool = True,
    ) -> torch.Tensor:
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                attend_previous=attend_previous,
            )
            * self.attn_gate.tanh()
            + x
        )
        residual_x = x
        for ff in self.feed_forward:
            x = ff(x)
        x = x * self.ff_gate.tanh() + residual_x

        return x


class OtterLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer: nn.Module, decoder_layer: nn.Module, use_dara: bool=False, use_only: bool=False, max_images: int=0):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Otter (https://github.com/dhansmair/otter-mini/)
    def condition_vis_x(self, vis_x) -> None:
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations) -> None:
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous) -> None:
        self.attend_previous = attend_previous

    def forward(
        self,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **decoder_layer_kwargs,
    ):
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
        )
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


class OtterLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_otter(
        self,
        media_token_id: int,
        vis_hidden_size: int,
        cross_attn_every_n_layers: int,
        use_media_placement_augmentation: bool,
        use_dara: bool=False,
        use_only: bool=False,
        max_images: int=0
    ):
        """
        Initialize Otter by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """

        if use_dara:
            gated_cross_attn_layers = []
            for layer_idx, _ in enumerate(self._get_decoder_layers()):
                if (layer_idx + 1) % cross_attn_every_n_layers == 0:
                    if layer_idx == 0:
                        gated_cross_attn_layers.append(
                            OtterGatedCrossAttentionBlock(
                                dim=self.config.hidden_size,
                                dim_visual=vis_hidden_size,
                                use_dara=use_dara,
                                use_only=use_only,
                                max_images=max_images,
                            )
                        )
                    else:
                        gated_cross_attn_layers.append(
                            OtterGatedCrossAttentionBlock(
                                dim=self.config.hidden_size,
                                dim_visual=vis_hidden_size,
                                use_dara=False,
                                use_only=use_only,
                                max_images=0,
                            )
                        )
                else:
                    gated_cross_attn_layers.append(None) 
            gated_cross_attn_layers = nn.ModuleList(gated_cross_attn_layers)
        
        else:
            gated_cross_attn_layers = nn.ModuleList(
            [
                OtterGatedCrossAttentionBlock(
                    dim=self.config.hidden_size,
                    dim_visual=vis_hidden_size,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
            )
        self._set_decoder_layers(nn.ModuleList([OtterLayer(gated_cross_attn_layer, decoder_layer) for gated_cross_attn_layer, decoder_layer in zip(gated_cross_attn_layers, self._get_decoder_layers())]))
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.initialized_otter = True

    def forward(self, *input, **kwargs):
        """Condition the Otter layers on the media locations before forward()"""
        if not self.initialized_otter:
            raise ValueError("Otter layers are not initialized. Please call `init_otter` first.")

        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0]
        media_locations = input_ids == self.media_token_id
        # IMPORTANT: Force `attend_previous` to True when we place training data as <image>caption<|endofchunk|>
        # attend_previous = (
        #     (random.random() < 0.5) if self.use_media_placement_augmentation else False
        # )
        attend_previous = (random.random() < 0.5) if self.use_media_placement_augmentation else True
        # attend_previous = self.only_attend_previous
        if self.__class__.__name__ == "My_LlamaForCausalLM":
            for layer in self.get_decoder().layers:
                layer.condition_media_locations(media_locations)
                layer.condition_attend_previous(attend_previous)
        elif self.__class__.__name__ == "LlamaForCausalLM":
            for layer in self.get_decoder().layers:
                layer.condition_media_locations(media_locations)
                layer.condition_attend_previous(attend_previous)
        elif self.__class__.__name__ in ["MPTForCausalLM", "MosaicGPT"]:
            for layer in self.get_decoder().blocks:
                layer.condition_media_locations(media_locations)
                layer.condition_attend_previous(attend_previous)
        else:
            master_print("inavaliable text encoder")
        return super().forward(*input, **kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self) -> None:
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_attend_previous(None)


class OtterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OtterConfig
    base_model_prefix = "otter"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OtterPerceiverBlock", "CLIPEncoderLayer", "OtterLayer", "CLIPVisionModel"]

    def _init_weights(self, module):
        """Otter requires no specific initialization"""
        return super()._init_weights(module)


class OtterModel(OtterPreTrainedModel):
    config_class = OtterConfig

    def __init__(
        self,
        config: OtterConfig, use_dara: bool=False, use_only: bool=False, max_images: int=0
    ):
        super().__init__(config)
        ### TODO: give "LlamaForCausalLM" as the name of text_config.architectures of Llama_based flamingo
        if "llama" not in config.text_config._name_or_path:
            if config.text_config.architectures[0] == "MPTForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct")
                lang_encoder = MPTForCausalLM(config=config.text_config)
            elif config.text_config.text_config.architectures[0] == "MosaicGPT":
                text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mosaic-llama-redpajama-final-candidate")
                lang_encoder = MosaicGPT(config=config.text_config)
            elif config.text_config.architectures[0] == "RWForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("PATH-TO-YOUR-FALCON")
                lang_encoder = RWForCausalLM(config=config.text_config)
        else:
            text_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path)
            if use_dara:
                lang_encoder = My_LlamaForCausalLM(config=config.text_config, use_dara=use_dara, max_images=max_images)
            if use_only:
                lang_encoder = My_LlamaForCausalLM(config=config.text_config, use_only=use_only)
            else:
                lang_encoder = LlamaForCausalLM(config=config.text_config)
        vision_encoder = CLIPVisionModel(config=config.vision_config)
        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]

        extend_instance(lang_encoder, OtterLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        if lang_encoder.__class__.__name__  == "My_LlamaForCausalLM":
            lang_encoder.resize_token_embeddings(len(text_tokenizer))
        elif lang_encoder.__class__.__name__ == "LlamaForCausalLM":
            lang_encoder.resize_token_embeddings(len(text_tokenizer))
        elif lang_encoder.__class__.__name__ == "My_LlamaForCausalLM":
            lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
        # use_media_placement_augmentation is strictly false for Otter model
        self.use_media_placement_augmentation = False  # config.use_media_placement_augmentation
        self.max_num_frames = config.max_num_frames if hasattr(config, "max_num_frames") else None

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = OtterPerceiverResampler(dim=self.vis_dim, max_num_frames=self.max_num_frames)

        self.lang_encoder.init_otter(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
            use_dara=use_dara, max_images=max_images
        )
        
        if "lora_config" in config.__dict__:
            master_print(f"Using LoRA with config:{config.lora_config}")
            standard_modules = ["q_proj", "v_proj"]
            lang_encoder_short_name = MODEL_CLASSES[config.text_config.architectures[0]]
            model_to_lora_modules = {
                "llama": standard_modules,
                "opt": standard_modules,
                "gptj": standard_modules,
                "gpt_neox": ["query_key_value"],
                "mpt": ["Wqkv"],
            }
            lora_config = LoraConfig(
                r=config.lora_config["r"],
                lora_alpha=config.lora_config["lora_alpha"],
                lora_dropout=config.lora_config["lora_dropout"],
                task_type=TaskType.CAUSAL_LM,
                target_modules=model_to_lora_modules[lang_encoder_short_name],
            )
            self.lang_encoder = get_peft_model(self.lang_encoder, lora_config)
            self.lang_encoder.master_print_trainable_parameters()

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.lang_encoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)

    def get_image_encoder(self) -> nn.Module:
        return self.vision_encoder

    def get_lang_encoder(self) -> nn.Module:
        return self.lang_encoder

    def tie_weights(self):
        return super().tie_weights()

    def init_weights(self):
        # Freeze all parameters in vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        for name, param in self.lang_encoder.named_parameters():
            if "gated_cross_attn_layer" not in name:
                param.requires_grad = False
        # Unfreeze LM input embeddings
        self.lang_encoder.get_input_embeddings().requires_grad_(True)
        ## MPTForCausalLM is tied word embedding
        if self.lang_encoder.__class__.__name__ == "LlamaForCausalLM":
            self.lang_encoder.lm_head.requires_grad_(True)
        # assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
        # master_print model size in billions of parameters in 2 decimal places
        master_print(f"Trainable param: {(sum(p.numel() for p in self.parameters() if p.requires_grad)) / 1e9:.2f} B")

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of Otter.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (vision_x is not None) or use_cached_vision_x, "Must provide either vision_x or use_cached_vision_x to True."

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert vision_x is None, "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        vision_x = self.vision_encoder(vision_x)[0][:, 1:, :]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)


class My_LlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config, use_dara=False, use_only=False, max_images=0):
        super().__init__(config)
        self.model = My_LlamaModel(config, use_dara=use_dara, use_only=use_only, max_images=max_images)
        

class OtterForConditionalGeneration(OtterPreTrainedModel):
    config_class = OtterConfig

    def __init__(
        self,
        config: OtterConfig, use_dara=False, use_only=False, max_images=0, attn_implementation=None
    ):
        super().__init__(config)
        ### TODO: give "LlamaForCausalLM" as the name of text_config.architectures of Llama_based flamingo
        if "llama" not in config.text_config._name_or_path:
            if config.text_config.architectures[0] == "MPTForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct")
                lang_encoder = MPTForCausalLM(config=config.text_config)
            elif config.text_config.architectures[0] == "MosaicGPT":
                text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mosaic-llama-redpajama-final-candidate")
                lang_encoder = MosaicGPT(config=config.text_config)
            elif config.text_config.architectures[0] == "RWForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained("PATH-TO-YOUR-FALCON")
                lang_encoder = RWForCausalLM(config=config.text_config)
            elif config.text_config.architectures[0] == "LlamaForCausalLM":
                text_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path)
                lang_encoder = LlamaForCausalLM(config=config.text_config)
            else:
                import pdb

                pdb.set_trace()
        else:
            text_tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path)
            if use_dara:
                lang_encoder = My_LlamaForCausalLM(config=config.text_config, use_dara=use_dara, max_images=max_images)
            elif use_dara:
                lang_encoder = My_LlamaForCausalLM(config=config.text_config, use_dara=use_only, max_images=max_images)
            else:
                lang_encoder = LlamaForCausalLM(config=config.text_config)
        vision_encoder = CLIPVisionModel(config=config.vision_config)

        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]

        extend_instance(lang_encoder, OtterLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        # if lang_encoder.__class__.__name__ == "LlamaForCausalLM":
        #     lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers
        # use_media_placement_augmentation is strictly false for Otter model
        self.use_media_placement_augmentation = False  # config.use_media_placement_augmentation
        self.max_num_frames = config.max_num_frames if hasattr(config, "max_num_frames") else None

        # Informative master_print statement
        if self.max_num_frames is None or self.max_num_frames == 1:
            master_print(f"The current model version is configured for Otter-Image with max_num_frames set to {self.max_num_frames}.")
        else:
            master_print(f"The current model version is configured for Otter-Video with a maximum of {self.max_num_frames} frames.")

        vision_encoder.output_tokens = True
        self.vision_encoder = vision_encoder

        self.vis_dim = 1024
        self.perceiver = OtterPerceiverResampler(dim=self.vis_dim, max_num_frames=self.max_num_frames)

        self.lang_encoder.init_otter(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            use_media_placement_augmentation=self.use_media_placement_augmentation,
            use_dara=use_dara, max_images=max_images
        )
        if "lora_config" in config.__dict__:
            original_architecture_name = self.lang_encoder.__class__.__name__
            master_print(f"Using LoRA with config:{config.lora_config}")
            standard_modules = ["q_proj", "v_proj"]
            lang_encoder_short_name = MODEL_CLASSES[config.text_config.architectures[0]]
            model_to_lora_modules = {
                "llama": standard_modules,
                "opt": standard_modules,
                "gptj": standard_modules,
                "gpt_neox": ["query_key_value"],
                "mpt": ["Wqkv"],
            }
            lora_config = LoraConfig(
                r=config.lora_config["r"],
                lora_alpha=config.lora_config["lora_alpha"],
                lora_dropout=config.lora_config["lora_dropout"],
                task_type=TaskType.CAUSAL_LM,
                target_modules=model_to_lora_modules[lang_encoder_short_name],
            )
            self.lang_encoder = get_peft_model(self.lang_encoder, lora_config)
            self.lang_encoder.master_print_trainable_parameters()
            self.lang_encoder.__class__.__name__ = f"{original_architecture_name}LoRA"

        self.post_init()
        self.use_only = use_only

    def get_input_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.lang_encoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)

    def get_image_encoder(self) -> nn.Module:
        return self.vision_encoder

    def get_lang_encoder(self) -> nn.Module:
        return self.lang_encoder

    def init_weights(self):
        # Freeze all parameters in self.model if train_vision_encoder is False or train_lang_encoder is False
        if not ("train_full_model" in self.config.__dict__ and self.config.train_full_model is True):
            for param in self.parameters():
                param.requires_grad = False

        # Freeze all parameters in vision encoder
        if "train_vision_encoder" in self.config.__dict__ and self.config.train_vision_encoder is True:
            master_print("Unfreeze vision encoder.")
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        if "train_lang_encoder" in self.config.__dict__ and self.config.train_lang_encoder is True:
            master_print("Unfreeze language decoder.")
            for name, param in self.lang_encoder.named_parameters():
                param.requires_grad = True

        # Freeze all parameters in vision encoder
        if "train_vision_encoder" in self.config.__dict__ and self.config.train_vision_encoder is True:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        if "train_lang_encoder" in self.config.__dict__ and self.config.train_lang_encoder is True:
            for name, param in self.lang_encoder.named_parameters():
                param.requires_grad = True

        # Freeze all parameters in vision encoder
        if "train_vision_encoder" in self.config.__dict__ and self.config.train_vision_encoder is True:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        if "train_lang_encoder" in self.config.__dict__ and self.config.train_lang_encoder is True:
            for name, param in self.lang_encoder.named_parameters():
                param.requires_grad = True

        if "lora_config" in self.config.__dict__:
            # Use another logic to unfreeze gated_cross_attn_layers and perceivers
            master_print(f"LoRA trainable param: {(sum(param.numel() for name, param in self.lang_encoder.named_parameters() if 'lora' in name)) / 1e6:.3f} M")
            for name, param in self.lang_encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

        # Freeze all parameters in lang encoders except gated_cross_attn_layers
        for name, param in self.lang_encoder.named_parameters():
            if "gated_cross_attn_layer" in name:
                param.requires_grad = True

        for name, param in self.named_parameters():
            if "perceiver" in name:
                param.requires_grad = True
        # Unfreeze LM input and output embeddings
        self.lang_encoder.get_input_embeddings().requires_grad_(True)
        ## MPTForCausalLM is tied word embedding
        if "LlamaForCausalLM" in self.lang_encoder.__class__.__name__:
            self.lang_encoder.lm_head.requires_grad_(True)

        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                master_print(f"Parameter: {name}, Size: {param.numel() / 1e6:.6f} M")
        master_print(f"Total Trainable param: {total_params / 1e9:.6f} B")

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        img_pos_ranges=None,
        input_ids=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of Otter.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (vision_x is not None) or use_cached_vision_x, "Must provide either vision_x or use_cached_vision_x to True."

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert vision_x is None, "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)
        
        img_pos_ranges = consecutive_ranges(lang_x[0], self.media_token_id)
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            img_pos_ranges=img_pos_ranges,
            **kwargs,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        vision_x = self.vision_encoder(vision_x)[0][:, 1:, :]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    @torch.no_grad()
    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if hasattr(self, "_hf_hook"):
            # add a hook to make sure that the output of lang_encoder is mapped to the same device as the lang_x
            hook = AlignDevicesHook(
                execution_device=lang_x.device,
                io_same_device=True,
                place_submodules=False,
            )
            add_hook_to_module(self.lang_encoder, hook)
        num_beams = generate_kwargs.get("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)
        self._encode_vision_x(vision_x=vision_x)
        
        if self.use_only:
            output, hidden_states_cd  = self.lang_encoder.generate(
                input_ids=lang_x,
                attention_mask=attention_mask,
                eos_token_id=self.eoc_token_id,
                **generate_kwargs,
            )
            hidden_states_cd = hidden_states_cd + 0.5 * outputs[0]
            logits_cd = self.lm_head(hidden_states_cd)
        else:
            output = self.lang_encoder.generate(
                input_ids=lang_x,
                attention_mask=attention_mask,
                eos_token_id=self.eoc_token_id,
                **generate_kwargs,
            )

        self.lang_encoder.clear_conditioned_layers()
        
        if self.use_only:
            return output, hidden_states_cd
        else:
            return output
            
