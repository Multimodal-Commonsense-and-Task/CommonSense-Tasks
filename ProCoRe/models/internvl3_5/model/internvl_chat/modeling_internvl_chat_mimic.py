import os
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union
import enum
import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from models.internvl3_5.conversation import get_conv_template
from models.internvl3_5.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from models.internvl3_5.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Attention, Qwen3DecoderLayer, Qwen3Model
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import TransformersKwargs
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn


logger = logging.get_logger(__name__)

class ShiftStrategy(enum.IntFlag):
    VECTOR_SHIFT = 1
    RECORD_HIDDEN_STATES = 4  # (not used in this minimal port)
    LEARNABLE_SHIFT_SCALE = 8
    MULTI_HEAD = 16


class MultiheadLinear(nn.Module):
    """
    Returns per-head scalar logits: [B,T,nh]
    Input: [B,T,nh,hd]
    """
    def __init__(self, num_heads: int, hidden_dim: int):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, head_dim, dtype=torch.bfloat16).normal_(0, 0.02))
        self.bias = nn.Parameter(torch.zeros([num_heads]))

    def forward(self, x):
        # x: [B,T,nh,hd]
        return torch.einsum("btnd,nd->btn", x, self.weight.to(x.device)) + self.bias.to(x.device)


class AttnApproxHandle:
    def __init__(self, active: bool = False):
        self.active = active

    def remove(self):
        self.active = False


class Qwen3VLAttnApproximator(nn.Module):
    """
    Code2.AttnApproximator port, specialized for Qwen3VL TextAttention.

    It injects a learnable shift AFTER attention output (pre o_proj),
    per layer (and optionally per head).
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.attn_strategy = attn_strategy

        # "handles" so you can enable/disable per layer like Code2
        self.attn_shift_handles = [AttnApproxHandle(active=False) for _ in range(self.num_layers)]

        # optional learnable mu scaling (Code2: LEARNABLE_SHIFT_SCALE)
        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
            if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                self.log_Z1_lin = nn.ModuleList(
                    MultiheadLinear(self.num_heads, self.hidden_dim) for _ in range(self.num_layers)
                )
            else:
                self.log_Z1_lin = nn.ModuleList(
                    nn.Linear(self.hidden_dim, 1) for _ in range(self.num_layers)
                )
        # shift parameters
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                # [L, nh, hd]
                self.attn_shift = nn.Parameter(
                    torch.randn(self.num_layers, self.num_heads, self.head_dim
                )* 0.001) 
            else:
                # [L, hidden]
                self.attn_shift = nn.Parameter(
                    torch.randn(self.num_layers, self.hidden_dim) * 0.001
                )

    def enable_all(self):
        for h in self.attn_shift_handles:
            h.active = True

    def disable_all(self):
        for h in self.attn_shift_handles:
            h.active = False

    def set_active(self, layer_idx: int, active: bool = True):
        self.attn_shift_handles[int(layer_idx)].active = active
        

    def do_shift(self, layer_idx: int, query_states, key_states, attn_output):
        """
        layer_idx: int
        query_states: [B, nh, T, hd]
        key_states:   [B, nh, K, hd]
        attn_output:  [B, T, nh, hd]  (matches eager_attention_forward output)
        returns:      [B, T, nh, hd]
        """
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return attn_output
        if not self.attn_shift_handles[layer_idx].active:
            return attn_output
        if not hasattr(self, "attn_shift"):
            return attn_output

        bsz, nh, T, hd = query_states.shape

        # Prepare shift in the same layout as attn_output [B,T,nh,hd]
        if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
            # shift: [1,1,nh,hd]
            shift = self.attn_shift[layer_idx][None, None, :, :]
        else:
            # shift: [1,1,hidden] -> reshape to [1,1,nh,hd]
            shift = self.attn_shift[layer_idx][None, None, :].view(1, 1, nh, hd)

        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
            # log_Z2: logsumexp(qk^T / sqrt(d)) across K  -> [B, nh, T]
            # [p40] query-chunked logsumexp (identical result, bounded memory: the full [B,nh,T,K] logits
            # matrix reaches >10 GB for ~15k-token inputs on a 24 GB GPU)
            _K = key_states.shape[-2]
            _step = max(1, 4_000_000 // max(1, _K))
            if T > _step:
                log_Z2 = torch.cat([
                    torch.logsumexp(torch.matmul(query_states[:, :, _i:_i + _step], key_states.transpose(-2, -1)) / (hd**0.5), dim=-1)
                    for _i in range(0, T, _step)
                ], dim=-1).transpose(-2, -1)  # [B, T, nh]
            else:
                log_Z2 = torch.logsumexp(
                    torch.matmul(query_states, key_states.transpose(-2, -1)) / (hd**0.5),
                    dim=-1,
                ).transpose(-2, -1)  # [B, T, nh]

            # compute log_Z1 from query states
            # query_states: [B, nh, T, hd] -> [B, T, nh, hd]
            q_t = query_states.transpose(1, 2)

            if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                log_Z1 = self.log_Z1_lin[layer_idx](q_t)  # [B,T,nh]
                mu = torch.exp(log_Z1 - torch.logaddexp(log_Z1, log_Z2))  # [B,T,nh]
                mu = mu.unsqueeze(-1)  # [B,T,nh,1]
            else:
                # reduce heads for log_Z2
                log_Z2_mean = log_Z2.mean(-1, keepdim=True)  # [B,T,1]
                # flatten q_t: [B,T,nh,hd] -> [B,T,hidden]
                q_flat = q_t.reshape(bsz, T, -1)
                _lin = self.log_Z1_lin[layer_idx]; log_Z1 = torch.nn.functional.linear(q_flat, _lin.weight.to(q_flat.device), None if _lin.bias is None else _lin.bias.to(q_flat.device))  # [B,T,1]
                mu = torch.exp(log_Z1 - torch.logaddexp(log_Z1, log_Z2_mean))  # [B,T,1]
                mu = mu.unsqueeze(-1)  # [B,T,1,1]
                # broadcast to heads
                mu = mu.expand(bsz, T, nh, 1)

            return attn_output + mu * shift.to(attn_output.device)

        # Simple vector shift (no mu)
        return attn_output + shift.to(attn_output.device)

# Copied from transformers.models.llama.modeling_llama.repeat_kv
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

def _eager_attention_forward_full(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    vis_token_amplifier: float = 0.0,
    dara_amplifier = None,
    use_dara: bool = False,
    **kwargs: Unpack[TransformersKwargs],
):
    img_pos_ranges = kwargs["img_pos_ranges"]
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # DARA (logits column scaling) — only if explicitly enabled
    if use_dara and img_pos_ranges is not None and len(img_pos_ranges) > 0:
        triples = []
        if isinstance(img_pos_ranges, (list, tuple)):
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

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # if vis_token_amplifier != 0.0 and len(img_pos_ranges)>0:
    #     # for start, end in img_pos_ranges:
    #     #     attn_weights[:, :, :, start : end + 1] *= self.vis_token_amplifier
    #     for start, end in img_pos_ranges:
    #         attn_weights[:, :, :, start: end+1] = (
    #         attn_weights[:, :, :, start: end+1].abs() * vis_token_amplifier
    #         + attn_weights[:, :, :, start: end+1]
    #         )
    #         # attn_weights[:, :, :, start: end+1] = attn_weights[:, :, :, start: end+1]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



# [p40] Query-chunked wrapper (same as modeling_internvl_chat.py): identical maths, never materialises the full
# [B,H,Q,K] logits. DARA/amplifier scale key columns only; the mask is sliced per query block.
_EAGER_CHUNK_Q = int(os.environ.get("P40_EAGER_CHUNK_Q", "1024"))
def eager_attention_forward(module, query, key, value, attention_mask, *args, **kwargs):
    B, H, Q, _ = query.shape
    K = key.shape[2]
    if _EAGER_CHUNK_Q <= 0 or Q <= _EAGER_CHUNK_Q or B * H * Q * K <= 64_000_000:
        return _eager_attention_forward_full(module, query, key, value, attention_mask, *args, **kwargs)
    outs = []
    for i in range(0, Q, _EAGER_CHUNK_Q):
        j = min(Q, i + _EAGER_CHUNK_Q)
        m = attention_mask[:, :, i:j, :] if attention_mask is not None else None
        o, _ = _eager_attention_forward_full(module, query[:, :, i:j], key, value, m, *args, **kwargs)
        outs.append(o)
    return torch.cat(outs, dim=1), None


def true_ranges(t):
    t = t.bool().cpu()
    padded = torch.cat([torch.tensor([False]), t, torch.tensor([False])])

    diff = padded[1:] ^ padded[:-1]

    change_idx = diff.nonzero(as_tuple=False).flatten()

    starts = change_idx[0::2]
    ends   = change_idx[1::2] - 1

    return list(zip(starts.tolist(), ends.tolist()))

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))




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



class My_Qwen3Attention(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int, use_dara=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config, layer_idx)
        if use_dara:
            # Amplifier used by DARA (select all heads by default)
            self._amp_head_idx = list(range(config.num_attention_heads))
            self.amplifier = AmplifiedAttentionHeadFactor(self._amp_head_idx, max_images=max_images)
        self.use_dara = use_dara
        self.vis_token_amplifier = vis_token_amplifier
        
        num_layers=config.num_hidden_layers
        hidden_dim=config.hidden_size
        self.num_heads=config.num_attention_heads
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
        attention_interface: Callable = eager_attention_forward
        if self.config.attn_implementation != "eager": # and not(self.use_dara or self.vis_token_amplifier is not 1.0)
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config.attn_implementation]

        if self.use_dara:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                dara_amplifier=self.amplifier,
                use_dara=self.use_dara,
                **kwargs,
            )
        elif self.vis_token_amplifier != 0.0:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                vis_token_amplifier = self.vis_token_amplifier,
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
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            
        parent_ref = self.__dict__.get("_parent_ref", None)
        parent = parent_ref() if parent_ref is not None else None
        attn_approx = getattr(parent, "attn_approx", None) if parent is not None else None

        if attn_approx is not None:
            # attn_output = attn_approx.do_shift(
            #     layer_idx=self.layer_idx,
            #     query_states=query_states,  # [B,nh,T,hd]
            #     key_states=key_states,      # [B,nh,K,hd] (K==T for prefill; larger with cache)
            #     attn_output=attn_output,    # [B,T,nh,hd]
            # )
            key_states_full = repeat_kv(key_states, self.num_key_value_groups)  # 8 -> 32
            assert query_states.shape[1] == self.num_heads
            assert key_states_full.shape[1] == query_states.shape[1]
            assert key_states_full.shape[-1] == query_states.shape[-1]
            attn_output = attn_approx.do_shift(
                layer_idx=self.layer_idx,
                query_states=query_states,      # [B, H, T, hd]
                key_states=key_states_full,     # [B, H, K, hd]
                attn_output=attn_output,        # [B, T, H, hd]
            )    
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_values

class My_Qwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int, use_dara=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config, layer_idx)
        self.self_attn = My_Qwen3Attention(config=config, layer_idx=layer_idx, use_dara=use_dara, max_images=max_images, vis_token_amplifier=vis_token_amplifier)
    
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attn_weights,present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hidden_states.shape[1] > 4096:  # [p40] token-chunked MLP for long prefills: exact, ~1GB less transient memory
            hidden_states = torch.cat([self.mlp(_h) for _h in hidden_states.split(2048, dim=1)], dim=1)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

class My_Qwen3Model(Qwen3Model):
    def __init__(self, config: Qwen3Config, use_dara=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config)
        
        if use_dara:
            self.layers = nn.ModuleList([
                
            ])
            self.layers = nn.ModuleList(
                [My_Qwen3DecoderLayer(
                    config, 
                    layer_idx, 
                    use_dara=use_dara if layer_idx == 0 else False, 
                    max_images=max_images if layer_idx == 0 else 0, 
                ) 
                for layer_idx in range(config.num_hidden_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [My_Qwen3DecoderLayer(config, layer_idx, vis_token_amplifier=vis_token_amplifier) for layer_idx in range(config.num_hidden_layers)]
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        all_self_attns = () if output_attentions else None
        output_hidden_states = kwargs.get("output_hidden_states", False)
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
                )
            
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1].cpu(),)
            if use_cache:
                next_decoder_cache = (layer_outputs[2 if output_attentions else 1])
            
        next_cache = next_decoder_cache if use_cache else None
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attentions=all_self_attns,
            hidden_states=all_hidden_states
        )
        
    @torch.no_grad()
    def add_mimic_module(
        self,
        enable_attn_shift: bool = True,
        attn_shift_multihead: bool = True,
        learnable_mu: bool = True,
        init_std: float = 0.02,
        shift_scale: float = 1e-3,
        keep_logZ1_fp32: bool = True,
        device: torch.device | None = None,
    ):
        """
        Call AFTER from_pretrained so we don't create params under meta init.

        - Creates self.attn_approx
        - Initializes params safely (fp32 init -> cast)
        - Wires _parent_ref for each attention layer
        """

        if not enable_attn_shift:
            self.attn_approx = None
            return

        # pick a real device:
        # If model was loaded with device_map, different layers may live on different GPUs.
        # For an LLM text stack, layer0's q_proj weight device is a good anchor.
        if device is None:
            try:
                device = next(self.layers[0].self_attn.q_proj.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # strategy flags
        strat = ShiftStrategy.VECTOR_SHIFT
        if attn_shift_multihead:
            strat |= ShiftStrategy.MULTI_HEAD
        if learnable_mu:
            strat |= ShiftStrategy.LEARNABLE_SHIFT_SCALE

        # create module on CPU first (or device), then init explicitly
        self.attn_approx = Qwen3VLAttnApproximator(
            num_layers=self.config.num_hidden_layers,
            hidden_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            attn_strategy=strat,
        )

        # ---- materialize + init weights safely (fp32 init -> cast) ----
        # 1) move to a real device (not meta)
        self.attn_approx.to(device=device)

        # 2) init attn_shift
        if hasattr(self.attn_approx, "attn_shift"):
            self.attn_approx.attn_shift.data = (
                torch.randn_like(self.attn_approx.attn_shift, dtype=torch.bfloat16, device=device) * shift_scale
            ).to(self.attn_approx.attn_shift.dtype)

        # 3) init log_Z1_lin weights/bias explicitly
        if hasattr(self.attn_approx, "log_Z1_lin"):
            for m in self.attn_approx.log_Z1_lin:
                if isinstance(m, MultiheadLinear):
                    m.weight.data = (torch.randn_like(m.weight, dtype=torch.bfloat16, device=device) * init_std).to(m.weight.dtype)
                    m.bias.data = torch.zeros_like(m.bias, dtype=torch.bfloat16, device=device).to(m.bias.dtype)
                elif isinstance(m, nn.Linear):
                    m.weight.data = (torch.randn_like(m.weight, dtype=torch.bfloat16, device=device) * init_std).to(m.weight.dtype)
                    if m.bias is not None:
                        m.bias.data = torch.zeros_like(m.bias, dtype=torch.bfloat16, device=device).to(m.bias.dtype)

            # optional: keep logZ1 in fp32 for stability
            if keep_logZ1_fp32:
                self.attn_approx.log_Z1_lin = self.attn_approx.log_Z1_lin.float()

        self.attn_approx.enable_all()

        # ---- wire parent refs so attention can find attn_approx ----
        import weakref
        for layer in self.layers:
            layer.self_attn.__dict__["_parent_ref"] = weakref.ref(self)

        # sanity check: make sure no meta params remain in attn_approx
        bad_meta = [n for n, p in self.attn_approx.named_parameters() if getattr(p, "is_meta", False)]
        if bad_meta:
            raise RuntimeError(f"attn_approx still has meta params: {bad_meta[:10]}")

class My_Qwen3ForCausalLM(Qwen3ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, use_dara=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config)
        self.model = My_Qwen3Model(config, use_dara=use_dara, max_images=max_images, vis_token_amplifier=vis_token_amplifier)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        img_pos_ranges=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            img_pos_ranges=img_pos_ranges,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class InternVLChatModel_mimic(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'Qwen3DecoderLayer', 'My_Qwen3DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True, use_dara=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen3ForCausalLM':
                self.language_model = My_Qwen3ForCausalLM(config.llm_config, use_dara=use_dara, max_images=max_images, vis_token_amplifier=vis_token_amplifier)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
        # Initialize weights and apply final processing. See: https://github.com/huggingface/transformers/pull/37708
        self.post_init()
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM','Qwen3ForCausalLM','My_Qwen3ForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplementedError
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.bfloat16, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
    
    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update(
            {
                'position_ids': position_ids,
                'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'),
                'attention_mask': attention_mask,
                'img_pos_ranges': kwargs.get('img_pos_ranges', None),
            }
        )
        return model_inputs

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             output_attentions=False, return_dict=False,
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output, img_pos_ind = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_attentions=output_attentions,
            **generation_config
        )
        if not output_attentions:
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split(template.sep.strip())[0].strip()
            history.append((question, response))
            if return_history:
                return response, history
            else:
                query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                if verbose:
                    print(query_to_print, response)
                return response
        else:
            return generation_output["attentions"], img_pos_ind

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        img_pos_ind = true_ranges(selected)
        # generate_kwargs["img_pos_ranges"] =img_pos_ind
        
        if output_attentions:
            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                use_cache=True,
                img_pos_ranges=img_pos_ind,
                **generate_kwargs,
            )
        else:
            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                use_cache=True,
                img_pos_ranges=img_pos_ind,
                **generate_kwargs,
            )
        return outputs, img_pos_ind

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update(
            {
                'position_ids': position_ids,
                'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'),
                'attention_mask': attention_mask,
                'img_pos_ranges': kwargs.get('img_pos_ranges', None),
            }
        )
        return model_inputs
    
    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
        