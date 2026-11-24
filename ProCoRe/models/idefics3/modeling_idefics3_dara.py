import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
# from ..utils.cache_utils import Cache, DynamicCache
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from transformers.models.idefics3.modeling_idefics3 import *
from transformers.models.idefics3.modeling_idefics3 import Idefics3BaseModelOutputWithPast, Idefics3CausalLMOutputWithPast, Idefics3Connector
from transformers.models.idefics3.configuration_idefics3 import *
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.nn import CrossEntropyLoss
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

import math
import copy
from functools import partial

logger = logging.get_logger(__name__)

IMAGE_TOKEN_ID = 0

import torch.nn.functional as F


def _normalize_img_pos_ranges(img_pos_ranges):
    if img_pos_ranges is None:
        return []

    if len(img_pos_ranges) > 0 and isinstance(img_pos_ranges[0], list):
        img_pos_ranges = img_pos_ranges[0]

    return img_pos_ranges


def scale_keys_for_dara(
    key_states,
    img_pos_ranges,
    dara_amplifier,
):
    """
    key_states: [B, H, K, D]

    DARA scales attention logits for image-token columns.
    Since logits = q @ k^T, scaling selected key vectors gives:
        q @ (alpha k)^T = alpha * q @ k^T
    """
    img_pos_ranges = _normalize_img_pos_ranges(img_pos_ranges)

    if dara_amplifier is None or len(img_pos_ranges) == 0:
        return key_states

    key_states = key_states.clone()

    head_idx = dara_amplifier.head_idx_tensor.to(key_states.device)
    amp = dara_amplifier.amplify_factor.to(
        device=key_states.device,
        dtype=key_states.dtype,
    )

    for slot, rng in enumerate(img_pos_ranges):
        if len(rng) == 2:
            s, e = int(rng[0]), int(rng[1])
            col = slot % dara_amplifier.max_images
        elif len(rng) == 3:
            s, e, raw_slot = int(rng[0]), int(rng[1]), int(rng[2])
            col = raw_slot % dara_amplifier.max_images
        else:
            raise ValueError("img_pos_ranges element must be (s,e) or (s,e,slot)")

        key_states[:, head_idx, s:e + 1, :] *= amp[:, col].view(1, -1, 1, 1)

    return key_states


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    dara_amplifier=None,
    use_dara=False,
    vis_token_amplifier=0.0,
    **kwargs,
):
    img_pos_ranges = kwargs.get("img_pos_ranges", None)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if use_dara:
        key_states = scale_keys_for_dara(
            key_states=key_states,
            img_pos_ranges=img_pos_ranges,
            dara_amplifier=dara_amplifier,
        )

    causal_mask = None
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None

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
    vis_token_amplifier = 0.0,
    **kwargs,
):
    img_pos_ranges = kwargs["img_pos_ranges"]
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if len(img_pos_ranges)>0 and isinstance(img_pos_ranges[0], list):
        img_pos_ranges = img_pos_ranges[0]
    
    # DARA (logits column scaling) — only if explicitly enabled
    if img_pos_ranges is not None and len(img_pos_ranges) > 0 and use_dara:
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
    if vis_token_amplifier != 0.0 and len(img_pos_ranges)>0:
    #     # for start, end in img_pos_ranges:
    #     #     attn_weights[:, :, :, start : end + 1] *= self.vis_token_amplifier
        for start, end in img_pos_ranges:
            attn_weights[:, :, -1, start: end+1] = (
            attn_weights[:, :, -1, start: end+1].abs() * vis_token_amplifier
            + attn_weights[:, :, -1, start: end+1]
            )

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    
    # if vis_token_amplifier != 1:
    #     for start, end in img_pos_ranges:
    #         attn_weights[:, :, :, start : end + 1] *= vis_token_amplifier
    
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
        if max_images > 0:
            # Learnable amplification matrix: [#selected_heads, max_images], init to 1.0
            self.amplify_factor = nn.Parameter(
                torch.ones(len(head_idx), max_images, dtype=torch.bfloat16),
                requires_grad=True,
            )
            self.max_images = max_images

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

    def __init__(self, config: LlamaConfig, layer_idx: int, use_dara: bool=False, max_images: int=0, vis_token_amplifier=0.0):
        super().__init__(config=config, layer_idx=layer_idx)
        self.num_heads = config.num_attention_heads
        self.use_dara = use_dara
        self.vis_token_amplifier = vis_token_amplifier
        # if use_dara:
        #     # Amplifier used by DARA (select all heads by default)
        #     self._amp_head_idx = list(range(self.num_heads))
        #     self.amplifier = AmplifiedAttentionHeadFactor(self._amp_head_idx, max_images=max_images)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # Always make sure img_pos_ranges exists (your code relied on it)
        img_pos_ranges = kwargs.get("img_pos_ranges", None)
        if img_pos_ranges is None:
            # your original logic (batch 0)
            org_input_ids = kwargs.get("org_input_ids", None)
            if org_input_ids is None:
                raise ValueError("img_pos_ranges is None and org_input_ids not provided.")
            img_pos_ranges = consecutive_ranges(org_input_ids[0], self.config.image_token_id)
            kwargs["img_pos_ranges"] = img_pos_ranges

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = sdpa_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation in ("sdpa",) or kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`scaled_dot_product_attention` doesn't support output_attentions=True. Falling back to eager."
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # ---------------------------
        # ✅ fetch DARA amplifier from parent model (injected later)
        # ---------------------------
        dara_amp = None
        if self.use_dara:
            parent_ref = self.__dict__.get("_parent_ref", None)
            parent = parent_ref() if parent_ref is not None else None
            dara_amp = getattr(parent, "dara_amplifier", None) if parent is not None else None

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            dara_amplifier=dara_amp,
            use_dara=(self.use_dara and dara_amp is not None),
            vis_token_amplifier=self.vis_token_amplifier,
            **kwargs,
        )
        # elif self.config._attn_implementation == "eager":
        #     attn_output, attn_weights = attention_interface(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask,
        #         dropout=0.0 if not self.training else self.attention_dropout,
        #         scaling=self.scaling,
        #         vis_token_amplifier=self.vis_token_amplifier,
        #         **kwargs,
        #     )
        # else:
        #     attn_output, attn_weights = attention_interface(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask,
        #         dropout=0.0 if not self.training else self.attention_dropout,
        #         scaling=self.scaling,
        #         **kwargs,
        #     )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class My_LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, use_dara: bool=False, max_images: int=0, vis_token_amplifier=0.0):
        super().__init__(config=config, layer_idx=layer_idx)

        self.self_attn = My_LlamaAttention(config=config, layer_idx=layer_idx, use_dara=use_dara, max_images=max_images, vis_token_amplifier=vis_token_amplifier)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class My_LlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, use_dara: bool=False, max_images: int=0, vis_token_amplifier=0.0):
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
        else:
            self.layers = nn.ModuleList(
                [My_LlamaDecoderLayer(config, layer_idx, vis_token_amplifier=vis_token_amplifier) for layer_idx in range(config.num_hidden_layers)]
            )

        # Initialize weights and apply final processing
        self.post_init()
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    
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
                    img_pos_ranges=img_pos_ranges,
                    org_input_ids=input_ids,
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
                    org_input_ids=input_ids,
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
    
    @torch.no_grad()
    def add_dara_module(
        self,
        max_images: int = 1,
        init_value: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        wire_parent_ref: bool = True,
    ):
        """
        Call AFTER from_pretrained() / after model is placed on real devices.

        Creates:
          self.dara_amplifier : AmplifiedAttentionHeadFactor

        Enables DARA by setting:
          self.layers[i].self_attn.use_dara = True

        Wires:
          layer.self_attn.__dict__["_parent_ref"] = weakref.ref(self)
        """
        layer_indices = [0]

        # Pick a real device anchored on an existing param
        if device is None:
            try:
                device = next(self.layers[int(layer_indices[0])].self_attn.q_proj.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        H = int(self.config.num_attention_heads)
        head_indices = list(range(H))

        # Create amplifier on a real device
        self.dara_amplifier = AmplifiedAttentionHeadFactor(head_indices, max_images=max_images).to(device=device)

        # Safe init (fp32 -> cast)
        if hasattr(self.dara_amplifier, "amplify_factor"):
            self.dara_amplifier.amplify_factor.data = (
                torch.full_like(self.dara_amplifier.amplify_factor, float(init_value),
                                dtype=torch.float32, device=device)
            ).to(dtype)

        # Enable DARA on specified layers
        for li in layer_indices:
            self.layers[int(li)].self_attn.use_dara = True

        # Wire weakref so attention can find self.dara_amplifier
        if wire_parent_ref:
            import weakref
            for layer in self.layers:
                layer.self_attn.__dict__["_parent_ref"] = weakref.ref(self)

        # Sanity: ensure no meta params
        bad_meta = [n for n, p in self.dara_amplifier.named_parameters() if getattr(p, "is_meta", False)]
        if bad_meta:
            raise RuntimeError(f"dara_amplifier still has meta params: {bad_meta[:10]}")



def consecutive_ranges(t: torch.Tensor, target_val: int):
    mask = (t == target_val)

    padded = torch.cat([torch.tensor([False], device=t.device),
                        mask,
                        torch.tensor([False], device=t.device)])

    diff = padded[1:] ^ padded[:-1]  # True where state changes

    change_idx = diff.nonzero(as_tuple=False).flatten()

    starts = change_idx[0::2]
    ends   = change_idx[1::2] - 1

    return list(zip(starts.tolist(), ends.tolist()))

class My_Idefics3Model(Idefics3Model):
    def __init__(self, config: Idefics3Config, use_dara=False, max_images=0., vis_token_amplifier=0.0):
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = Idefics3VisionTransformer._from_config(config.vision_config)
        self.connector = Idefics3Connector(config)
        self.text_model = My_LlamaModel(config.text_config, use_dara=use_dara, max_images=max_images, vis_token_amplifier=vis_token_amplifier)
        
        self.use_dara = use_dara
        
        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

        self._use_flash_attention_2 = config.text_config._attn_implementation == "flash_attention_2"
        self.post_init()
        self._cached_img_pos_ranges = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        img_pos_ranges: Optional[object] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if (cache_position is not None and cache_position[0] == 0) and img_pos_ranges is None:
            img_pos_ranges = [consecutive_ranges(input_id, self.config.image_token_id) for input_id in input_ids]
            if isinstance(img_pos_ranges[0], list):
                img_pos_ranges = img_pos_ranges[0]
            self._cached_img_pos_ranges = img_pos_ranges
        elif (cache_position is not None and cache_position[0] == 0):
            self._cached_img_pos_ranges = img_pos_ranges
        else:
            img_pos_ranges = getattr(self, "_cached_img_pos_ranges", None)
        
        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
            img_pos_ranges=img_pos_ranges,
        )

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return Idefics3BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class My_Idefics3ForConditionalGeneration_dara(Idefics3ForConditionalGeneration):
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.__init__ with Idefics2->Idefics3
    def __init__(self, config, use_dara=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config)
        self.model = My_Idefics3Model(config, use_dara=use_dara, max_images=max_images, vis_token_amplifier=vis_token_amplifier)
        self.image_token_id = self.config.image_token_id

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()
        
        IMAGE_TOKEN_ID = self.image_token_id
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        img_pos_ranges=None,
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics3ForConditionalGeneration`).
                Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
                computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).
        Returns:

        Example:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModelForVision2Seq
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
        >>> model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16, device_map="auto")

        >>> # Create inputs
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image"},
        ...             {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
        ...             {"type": "image"},
        ...             {"type": "text", "text": "What can we see in this image?"},
        ...         ]
        ...     },
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image"},
        ...             {"type": "text", "text": "In which city is that bridge located?"},
        ...         ]
        ...     }
        ... ]

        >>> prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
        >>> images = [[image1, image2], [image3]]
        >>> inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts[0])
        Assistant: There are buildings, trees, lights, and water visible in this image.

        >>> print(generated_texts[1])
        Assistant: The bridge is in San Francisco.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
            img_pos_ranges=img_pos_ranges
        )
 
        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Idefics3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
