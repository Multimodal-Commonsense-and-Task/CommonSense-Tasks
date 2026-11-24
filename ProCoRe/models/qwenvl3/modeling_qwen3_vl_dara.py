import os
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from transformers.models.qwen3_vl.modeling_qwen3_vl import *
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, Qwen3VLTextDecoderLayer, Qwen3VLTextModel, Qwen3VLModel, Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import *
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.masking_utils import create_causal_mask
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import math
import copy

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def keep_only_last_true_run(x: torch.Tensor) -> torch.Tensor:
    """
    Keep only the last consecutive True-run.
    Works for:
      - python list of bools length L
      - torch.BoolTensor shape [L] or [B, L]
    Returns same "kind" (torch tensor).
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.bool)

    x = x.bool()
    orig_1d = (x.ndim == 1)
    if orig_1d:
        x = x.unsqueeze(0)  # [1, L]

    B, L = x.shape
    device = x.device

    any_true = x.any(dim=1)  # [B]
    idx = torch.arange(L, device=device).view(1, L)  # [1, L]

    # last True index per row, -1 if no True
    last_true = (x.long() * idx).amax(dim=1)  # [B]
    last_true = torch.where(any_true, last_true, last_true.new_full((B,), -1))

    up_to_last = idx <= last_true.view(B, 1)  # [B, L]

    # ignore region after last_true by forcing True there
    y = x | (~up_to_last)

    # contiguous-to-the-right check up to last_true
    right_all = torch.cumprod(y.flip(dims=(1,)).long(), dim=1).flip(dims=(1,)).bool()

    out = x & up_to_last & right_all

    return out.squeeze(0) if orig_1d else out
    

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


def remove_visual_from_input_ids(input_ids: torch.Tensor, image_spans_per_sample):
    """
    input_ids: LongTensor [B, L]
    image_spans_per_sample: List[List[Tuple[int,int]]] length B
        each span is (start, end) with an inclusive end: [start, end]
    Returns:
      new_input_ids: LongTensor [B, L_new_max] (padded)
      new_attention_mask: BoolTensor [B, L_new_max]
      lengths: LongTensor [B]
    """
    B, L = input_ids.shape
    device = input_ids.device

    kept_seqs = []
    lengths = []

    is_batched = (
        len(image_spans_per_sample) == B
        and all(
            isinstance(ranges_b, (list, tuple))
            and (
                len(ranges_b) == 0
                or isinstance(ranges_b[0], (list, tuple))
            )
            for ranges_b in image_spans_per_sample
        )
    )
    per_batch_ranges = (
        image_spans_per_sample
        if is_batched
        else [image_spans_per_sample] * B
    )

    for b, ranges_b in enumerate(per_batch_ranges):
        mask = torch.ones(L, dtype=torch.bool, device=device)
        for image_range in ranges_b:
            s, e = image_range[:2]
            mask[s:e + 1] = False

        kept = input_ids[b][mask]          # [L_b]
        kept_seqs.append(kept)
        lengths.append(kept.numel())

    lengths = torch.tensor(lengths, device=device, dtype=torch.long)

    # pad to rectangular [B, max_len]
    pad_id = 0  # change if your tokenizer uses something else (e.g., tokenizer.pad_token_id)
    new_input_ids = pad_sequence(kept_seqs, batch_first=True, padding_value=pad_id)
    new_attention_mask = torch.arange(new_input_ids.size(1), device=device)[None, :] < lengths[:, None]

    return new_input_ids, new_attention_mask, lengths


def _eager_attention_forward_full(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    dara_amplifier = None,
    use_dara = False,
    vis_token_amplifier: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):

    img_pos_ranges = kwargs.get("img_pos_ranges", None)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # DARA (logits column scaling) — only if explicitly enabled. Keep ranges
    # grouped by batch item so every image gets its own amplifier slot.
    if use_dara and img_pos_ranges is not None and len(img_pos_ranges) > 0:
        is_batched = (
            len(img_pos_ranges) == attn_weights.shape[0]
            and all(
                isinstance(ranges_b, (list, tuple))
                and (
                    len(ranges_b) == 0
                    or isinstance(ranges_b[0], (list, tuple))
                )
                for ranges_b in img_pos_ranges
            )
        )
        per_batch_ranges = img_pos_ranges if is_batched else [img_pos_ranges]
        if len(per_batch_ranges) != attn_weights.shape[0]:
            raise ValueError(
                "img_pos_ranges batch size does not match attention batch size: "
                f"{len(per_batch_ranges)} != {attn_weights.shape[0]}"
            )

        triples = []
        for batch_index, ranges_b in enumerate(per_batch_ranges):
            next_slot = 0
            for rng in ranges_b:
                if len(rng) == 2:
                    s, e = int(rng[0]), int(rng[1])
                    triples.append((batch_index, s, e, next_slot))
                    next_slot += 1
                elif len(rng) == 3:
                    s, e, slot = int(rng[0]), int(rng[1]), int(rng[2])
                    triples.append((batch_index, s, e, slot))
                else:
                    raise ValueError(
                        "img_pos_ranges element must be (s,e) or (s,e,slot)"
                    )

        if triples:
            img_pos_tensor = torch.as_tensor(
                triples, device=attn_weights.device, dtype=torch.long
            )
            finite_mask = torch.isfinite(attn_weights)  # defensive
            attn_weights = dara_amplifier(attn_weights, img_pos_tensor)
            attn_weights = torch.where(finite_mask, attn_weights, attn_weights.new_full((), float("-inf")))
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
        
    if vis_token_amplifier != 0.0 and img_pos_ranges is not None and len(img_pos_ranges)>0:
        is_batched = (
            len(img_pos_ranges) == attn_weights.shape[0]
            and all(
                isinstance(ranges_b, (list, tuple))
                and (
                    len(ranges_b) == 0
                    or isinstance(ranges_b[0], (list, tuple))
                )
                for ranges_b in img_pos_ranges
            )
        )
        per_batch_ranges = img_pos_ranges if is_batched else [img_pos_ranges]
        if len(per_batch_ranges) != attn_weights.shape[0]:
            raise ValueError(
                "img_pos_ranges batch size does not match attention batch size: "
                f"{len(per_batch_ranges)} != {attn_weights.shape[0]}"
            )
        for batch_index, ranges_b in enumerate(per_batch_ranges):
            for image_range in ranges_b:
                start, end = image_range[:2]
                image_logits = attn_weights[
                    batch_index, :, :, start:end + 1
                ]
                attn_weights[batch_index, :, :, start:end + 1] = (
                    image_logits.abs() * vis_token_amplifier + image_logits
                )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# [p40] Query-chunked wrapper: numerically identical to the full eager path (DARA / amplifier scale key
# columns only, the mask is sliced per query block), but never materialises the full [B,H,Q,K] logits.
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

class AmplifiedAttentionHeadFactor(nn.Module):
    def __init__(self, attention_amplify_head_ind, max_images: int = 1):
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

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
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
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class My_Qwen3VLTextAttention(Qwen3VLTextAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, use_dara: bool=False, use_pai: bool=False, max_images: int=0, vis_token_amplifier=0.0):
        super().__init__(config, layer_idx)
        self.use_dara = use_dara
        self.use_pai = use_pai
        self.num_heads = config.num_attention_heads
        self.vis_token_amplifier = vis_token_amplifier
        # if use_dara:
        #     # Amplifier used by DARA (select all heads by default)
        #     self._amp_head_idx = list(range(self.num_heads))
        #     self.amplifier = AmplifiedAttentionHeadFactor(self._amp_head_idx, max_images=max_images)
            
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
        
        output_attentions = kwargs.get("output_attentions", False)
            
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
        dara_amp = None
        if self.use_dara:
            parent_ref = self.__dict__.get("_parent_ref", None)
            parent = parent_ref() if parent_ref is not None else None
            dara_amp = getattr(parent, "dara_amplifier", None) if parent is not None else None

        if self.use_dara and dara_amp is not None:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                dara_amplifier=dara_amp,
                use_dara=True,
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
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights= None
        
        return attn_output, attn_weights
        


class My_Qwen3VLTextDecoderLayer(Qwen3VLTextDecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, use_dara: bool=False, use_pai: bool=False, max_images: int=0, vis_token_amplifier: float=0.0):
        super().__init__(config, layer_idx)

        self.self_attn = My_Qwen3VLTextAttention(config=config, layer_idx=layer_idx, use_dara=use_dara, use_pai=use_pai, max_images=max_images, vis_token_amplifier=vis_token_amplifier)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        output_attentions = kwargs.get("output_attentions", False)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weight = self.self_attn(
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
        
        if output_attentions:
            return hidden_states, self_attn_weight
        
        return hidden_states
    
class My_Qwen3VLTextModel(Qwen3VLTextModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig, use_dara=False, use_pai=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config)
        
        if use_dara:
            self.layers = nn.ModuleList(
                [My_Qwen3VLTextDecoderLayer(config, 
                                            layer_idx, 
                                            use_dara=use_dara if layer_idx == 0 else False,
                                            max_images=max_images if layer_idx == 0 else 0,
                                            ) 
                for layer_idx in range(config.num_hidden_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [My_Qwen3VLTextDecoderLayer(config, layer_idx, use_pai=use_pai, vis_token_amplifier=vis_token_amplifier) for layer_idx in range(config.num_hidden_layers)]
            )

            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        output_attentions = kwargs.get("output_attentions", False)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        all_self_attns = () if output_attentions else None
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_attentions:
                layer_outputs, attn_weights = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                all_self_attns += (attn_weights,)
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        if output_attentions:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                attentions=all_self_attns,
            )
            
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
            )
    @torch.no_grad()
    def add_dara_module(
        self,
        max_images: int = 1,
        init_value: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.bfloat16,
        wire_parent_ref: bool = True,
    ):
        """
        Call AFTER from_pretrained() to avoid meta params.

        Creates:
          self.dara_amplifier: AmplifiedAttentionHeadFactor

        And enables DARA on specified layers by setting:
          layer.self_attn.use_dara = True

        Also wires _parent_ref so attention can find self.dara_amplifier.
        """

        layer_indices = [0]

        # pick device anchor
        if device is None:
            try:
                device = next(self.layers[layer_indices[0]].self_attn.q_proj.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # heads
        H = self.config.num_attention_heads
        head_indices = list(range(H))

        # create amplifier and materialize on real device
        self.dara_amplifier = AmplifiedAttentionHeadFactor(head_indices, max_images=max_images).to(device=device)

        # init amplify_factor
        if hasattr(self.dara_amplifier, "amplify_factor"):
            # init in fp32 then cast (safer)
            self.dara_amplifier.amplify_factor.data = (
                torch.full_like(self.dara_amplifier.amplify_factor, float(init_value), dtype=torch.float32, device=device)
            ).to(dtype if dtype is not None else self.dara_amplifier.amplify_factor.dtype)

        # enable DARA on chosen layers
        for li in layer_indices:
            self.layers[int(li)].self_attn.use_dara = True

        # wire weakref parent so attention can find parent.dara_amplifier
        if wire_parent_ref:
            import weakref
            for layer in self.layers:
                layer.self_attn.__dict__["_parent_ref"] = weakref.ref(self)

        # meta sanity
        bad_meta = [n for n, p in self.dara_amplifier.named_parameters() if getattr(p, "is_meta", False)]
        if bad_meta:
            raise RuntimeError(f"dara_amplifier still has meta params: {bad_meta[:10]}")
        

class My_Qwen3VLModel(Qwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "My_Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config, use_dara=False, use_pai=False, use_icot=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config)
        self.language_model = My_Qwen3VLTextModel._from_config(config.text_config, use_dara=use_dara, use_pai=use_pai, max_images=max_images, vis_token_amplifier=vis_token_amplifier)

        # Initialize weights and apply final processing
        self.post_init()
        self.use_icot = use_icot
        self.num_line_break = 0
        self.num_sub_imgs = 0
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        output_attentions = kwargs.get("output_attentions", False)
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.use_icot:
            judge = input_ids[:, -1] in [382] or input_ids[:, -1] in [624]
            if judge:
                self.num_line_break += 1

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                self.num_line_break = 0
                self.num_sub_imgs = 0

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if self.use_icot:
            if pixel_values is not None:
                self.reasoning_img_embeds = image_embeds[-(image_grid_thw[:, 1] * image_grid_thw[:, 2] // 4)[-1]:, ...]
                self.query_image_mask = keep_only_last_true_run(visual_pos_masks[0])
                if len(self.query_image_mask.shape) == 1:
                    self.query_image_mask = self.query_image_mask.unsqueeze(dim=0)
        
            if self.num_line_break and self.num_line_break % 2 == 0 and self.num_sub_imgs < 3:
                tmp_copy_pkv = copy.deepcopy(past_key_values)
                with torch.no_grad():
                    outputs = self.language_model(
                                    input_ids=None,
                                    past_key_values=tmp_copy_pkv,
                                    inputs_embeds=inputs_embeds,
                                    use_cache=False,
                                    output_attentions=True,
                                )
                del tmp_copy_pkv
                image_attentions = torch.cat(outputs.attentions, dim=1).mean(dim=1)[:, -1]
                if self.query_image_mask.shape[-1] != image_attentions.shape[-1]:
                    self.query_image_mask = torch.cat([self.query_image_mask, torch.zeros(self.query_image_mask.shape[0],
                                                                                        image_attentions.shape[-1] - self.query_image_mask.shape[-1],
                                                                                        device=self.query_image_mask.device).bool()],
                                                                                        dim=1)
                image_attentions = image_attentions[self.query_image_mask]
                indices = image_attentions.topk(16)[1].sort()[0]
                sampled_reasoning_embeds = self.reasoning_img_embeds[indices]
                x_ids = torch.tensor([151652] + [151655]*16 + [151653], device=input_ids.device).unsqueeze(0)
                x_embeds = self.language_model.embed_tokens(x_ids)
                x_embeds = x_embeds.masked_scatter((x_ids == 151655).unsqueeze(-1).expand_as(x_embeds), sampled_reasoning_embeds.unsqueeze(0))
                inputs_embeds = torch.cat([inputs_embeds, x_embeds], dim=1)
                interleaved_position_ids = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5], 
                                                        [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5], 
                                                        [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5]], device=input_ids.device).unsqueeze(1)
                position_ids = torch.cat([position_ids, interleaved_position_ids + position_ids.max() + 1], dim=-1)
                self.rope_deltas += -12
                self.num_sub_imgs += 1
                cache_position += 18*(self.num_sub_imgs-1)
                cache_position = torch.cat([cache_position, torch.arange(18, device=cache_position.device) + cache_position[-1]+1], dim=0)
                
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], self.num_sub_imgs*18), device=attention_mask.device)], dim=-1)
        
        
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        if output_attentions:
            return Qwen3VLModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                rope_deltas=self.rope_deltas,
                attentions=outputs.attentions
            )
        else:
            return Qwen3VLModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                rope_deltas=self.rope_deltas,
            )
        
    

class My_Qwen3VLForConditionalGeneration_dara(Qwen3VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config, use_dara=False, use_pai=False, use_icot=False, max_images=0, vis_token_amplifier=0.0):
        super().__init__(config)
        self.model = My_Qwen3VLModel(config, use_dara=use_dara, use_pai=use_pai, use_icot=use_icot, max_images=max_images, vis_token_amplifier=vis_token_amplifier)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.use_pai = use_pai
        self.post_init()
        self._cached_img_pos_ranges = None
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        
        is_initial_pass = input_ids is not None and (
            cache_position is None or cache_position[0] == 0
        )
        if is_initial_pass:
            img_pos_ranges = [consecutive_ranges(input_id, self.config.image_token_id) for input_id in input_ids]
            self._cached_img_pos_ranges = img_pos_ranges
        else:
            img_pos_ranges = getattr(self, "_cached_img_pos_ranges", None)
        if self.use_pai:
            new_ids, new_mask, lens = remove_visual_from_input_ids(input_ids, img_pos_ranges)
            
        kwargs["img_pos_ranges"] = img_pos_ranges
        output_attentions = kwargs.get("output_attentions", False)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if self.use_pai:
            uncondition_output = self.model(new_ids, use_cache=True)
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            unconditional_logits = self.lm_head(hidden_states[:, slice_indices, :])
            cutoff = torch.log(torch.tensor(0.1)) + logits.max(dim=-1, keepdim=True).values
            out = (
                0.5 * (logits - unconditional_logits) + unconditional_logits
            )
            logits = out.masked_fill(logits < cutoff, -float("inf"))

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        if output_attentions:
            return Qwen3VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                rope_deltas=outputs.rope_deltas,
                attentions=outputs.attentions
            )
        else:
            return Qwen3VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                rope_deltas=outputs.rope_deltas,
            )
