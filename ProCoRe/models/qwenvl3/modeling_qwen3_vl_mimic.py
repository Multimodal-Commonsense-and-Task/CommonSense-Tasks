import enum
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import copy
import math
import weakref

from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.masking_utils import create_causal_mask

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLConfig


# ============================================================
#  Shift strategy + Approximator (Code2-style "AttnApproximator")
# ============================================================

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

    # def do_shift(self, layer_idx: int, query_states, key_states, attn_output):
    #     layer_idx = int(layer_idx)
    #     if layer_idx < 0 or layer_idx >= self.num_layers:
    #         return attn_output
    #     if not self.attn_shift_handles[layer_idx].active:
    #         return attn_output
    #     if not hasattr(self, "attn_shift"):
    #         return attn_output

    #     B, nh, T, hd = query_states.shape

    #     # shift -> [1,1,nh,hd]
    #     if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
    #         shift = self.attn_shift[layer_idx][None, None, :, :]                 # [1,1,nh,hd]
    #     else:
    #         shift = self.attn_shift[layer_idx][None, None, :].view(1, 1, nh, hd) # [1,1,nh,hd]

    #     if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
    #         # ---- compute log_Z2 in float32 for stability ----
    #         q = query_states.float()  # [B,nh,T,hd]
    #         k = key_states.float()    # [B,nh,K,hd]
    #         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)  # [B,nh,T,K]
    #         log_Z2 = torch.logsumexp(scores, dim=-1).transpose(-2, -1)     # [B,T,nh]

    #         # ---- compute log_Z1 in float32 ----
    #         q_t = query_states.transpose(1, 2).float()  # [B,T,nh,hd]

    #         if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
    #             log_Z1 = self.log_Z1_lin[layer_idx](q_t).float()  # [B,T,nh]
    #             delta = log_Z1 - log_Z2                           # [B,T,nh]
    #             delta = delta.clamp(-20, 20)
    #             mu = torch.sigmoid(delta).unsqueeze(-1)           # [B,T,nh,1]
    #         else:
    #             log_Z2m = log_Z2.mean(-1, keepdim=True)           # [B,T,1]
    #             q_flat = q_t.reshape(B, T, -1)
    #             log_Z1 = self.log_Z1_lin[layer_idx](q_flat).float()  # [B,T,1]
    #             delta = (log_Z1 - log_Z2m).clamp(-20, 20)
    #             mu = torch.sigmoid(delta).unsqueeze(-1).expand(B, T, nh, 1)  # [B,T,nh,1]

    #         # ---- sanitize ----
    #         mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=0.0).to(attn_output.dtype)
    #         shift = shift.to(attn_output.dtype)

    #         return attn_output + mu * shift

    #     return attn_output + shift.to(attn_output.dtype)


# ============================================================
#  Helpers (your original helpers, with fixes)
# ============================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def keep_only_last_true_run(x: torch.Tensor) -> torch.Tensor:
    """
    Keep only the last consecutive True-run.
    Works for:
      - python list of bools length L
      - torch.BoolTensor shape [L] or [B, L]
    Returns torch tensor.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def remove_visual_from_input_ids(input_ids: torch.Tensor, image_spans_per_sample):
    """
    input_ids: LongTensor [B, L]
    image_spans_per_sample: List[List[Tuple[int,int]]] length B
        each span is (start, end) with end exclusive: [start, end)
    Returns:
      new_input_ids: LongTensor [B, L_new_max] (padded)
      new_attention_mask: BoolTensor [B, L_new_max]
      lengths: LongTensor [B]
    """
    B, L = input_ids.shape
    device = input_ids.device

    kept_seqs = []
    lengths = []

    for b in range(B):
        mask = torch.ones(L, dtype=torch.bool, device=device)
        for s, e in image_spans_per_sample[b]:   # ✅ FIX
            mask[int(s):int(e)] = False

        kept = input_ids[b][mask]
        kept_seqs.append(kept)
        lengths.append(kept.numel())

    lengths = torch.tensor(lengths, device=device, dtype=torch.long)

    pad_id = 0  # set to tokenizer.pad_token_id if you want
    new_input_ids = pad_sequence(kept_seqs, batch_first=True, padding_value=pad_id)
    new_attention_mask = torch.arange(new_input_ids.size(1), device=device)[None, :] < lengths[:, None]

    return new_input_ids, new_attention_mask, lengths


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def consecutive_ranges(t: torch.Tensor, target_val: int):
    mask = (t == target_val)

    padded = torch.cat([
        torch.tensor([False], device=t.device),
        mask,
        torch.tensor([False], device=t.device),
    ])

    diff = padded[1:] ^ padded[:-1]
    change_idx = diff.nonzero(as_tuple=False).flatten()

    starts = change_idx[0::2]
    ends   = change_idx[1::2] - 1

    return list(zip(starts.tolist(), ends.tolist()))


# ============================================================
#  DARA helper (your original; unchanged except minor safety)
# ============================================================

class AmplifiedAttentionHeadFactor(nn.Module):
    def __init__(self, attention_amplify_head_ind, max_images: int = 1):
        super().__init__()
        head_idx = list(map(int, attention_amplify_head_ind))
        self.register_buffer(
            "head_idx_tensor",
            torch.as_tensor(head_idx, dtype=torch.long),
            persistent=False,
        )
        self.max_images = max_images

        if max_images > 0:
            self.amplify_factor = nn.Parameter(
                torch.ones(len(head_idx), max_images),
                requires_grad=True,
            )

    @torch.no_grad()
    def _interval_mask_single(self, B, K, b: int, s: int, e: int, device, dtype):
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
        """
        B, H, Q, K = attention_scores.shape
        device = attention_scores.device
        dtype  = attention_scores.dtype

        if img_pos is None or img_pos.numel() == 0:
            return attention_scores

        img_pos = img_pos.to(device=device)
        if img_pos.size(1) != 4:
            raise ValueError("img_pos must be [N, 4] = (batch_idx, start, end, slot).")

        combined = torch.ones((B, H, 1, K), device=device, dtype=dtype)
        heads_all_ones = torch.ones((H,), device=device, dtype=dtype)

        for i in range(img_pos.size(0)):
            b = int(img_pos[i, 0].item())
            s = int(img_pos[i, 1].item())
            e = int(img_pos[i, 2].item())
            raw_slot = int(img_pos[i, 3].item())
            col = raw_slot % self.max_images

            mask_row = self._interval_mask_single(B, K, b, s, e, device, dtype)

            scales_H = heads_all_ones.clone()
            if self.head_idx_tensor.numel() > 0:
                learnable_col = self.amplify_factor[:, col].to(dtype=dtype)
                scales_H.scatter_(0, self.head_idx_tensor, learnable_col)

            delta = (scales_H - 1.0).view(1, H, 1, 1)
            combined = combined + delta * mask_row

        attention_scores.mul_(combined)
        return attention_scores


# ============================================================
#  Eager attention forward (your original; kept)
# ============================================================

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    dara_amplifier=None,
    use_dara: bool = False,
    vis_token_amplifier: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    img_pos_ranges = kwargs.get("img_pos_ranges", None)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # DARA
    if use_dara and img_pos_ranges is not None and len(img_pos_ranges) > 0:
        triples = []
        next_slot = 0
        for rng in img_pos_ranges:
            if len(rng) == 2:
                s, e = int(rng[0]), int(rng[1])
                triples.append((0, s, e, next_slot))
                next_slot += 1
            elif len(rng) == 3:
                s, e, sl = int(rng[0]), int(rng[1]), int(rng[2])
                triples.append((0, s, e, sl))
            else:
                raise ValueError("img_pos_ranges element must be (s,e) or (s,e,slot)")
        img_pos_tensor = torch.as_tensor(triples, device=attn_weights.device, dtype=torch.long)
        finite_mask = torch.isfinite(attn_weights)
        attn_weights = dara_amplifier(attn_weights, img_pos_tensor)
        attn_weights = torch.where(finite_mask, attn_weights, attn_weights.new_full((), float("-inf")))

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if vis_token_amplifier != 0.0 and img_pos_ranges is not None and len(img_pos_ranges) > 0:
        for start, end in img_pos_ranges:
            attn_weights[:, :, :, start : end + 1] = (
                attn_weights[:, :, :, start : end + 1].abs() * vis_token_amplifier
                + attn_weights[:, :, :, start : end + 1]
            )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B,T,nh,hd]
    return attn_output, attn_weights

class My_Qwen3VLTextAttention(Qwen3VLTextAttention):
    def __init__(
        self,
        config: Qwen3VLTextConfig,
        layer_idx: int,
        use_dara: bool = False,
        use_pai: bool = False,
        max_images: int = 0,
        vis_token_amplifier: float = 0.0,
    ):
        super().__init__(config, layer_idx)
        self.use_dara = use_dara
        self.use_pai = use_pai
        self.num_heads = config.num_attention_heads
        self.vis_token_amplifier = vis_token_amplifier

        # pointer set by My_Qwen3VLTextModel after layer construction
        # self.parent_model = None

        if use_dara:
            self._amp_head_idx = list(range(self.num_heads))
            self.amplifier = AmplifiedAttentionHeadFactor(self._amp_head_idx, max_images=max_images)

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

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,nh,T,hd]
        key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,nh,T,hd]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)               # [B,nh,T,hd]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.use_dara:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                dara_amplifier=self.amplifier,
                use_dara=True,
                **kwargs,
            )
        elif self.vis_token_amplifier != 0.0:  # ✅ FIX (was `is not 0.0`)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                vis_token_amplifier=self.vis_token_amplifier,
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
        # ------------------------------------------------------------
        # ✅ Code2-style: inject shift AFTER attention output, pre o_proj
        # attn_output: [B,T,nh,hd]
        # ------------------------------------------------------------
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

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class My_Qwen3VLTextDecoderLayer(Qwen3VLTextDecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, use_dara: bool = False, use_pai: bool = False,
                 max_images: int = 0, vis_token_amplifier: float = 0.0):
        super().__init__(config, layer_idx)

        # ✅ IMPORTANT: self_attn must be an attention module, NOT the approximator
        self.self_attn = My_Qwen3VLTextAttention(
            config=config,
            layer_idx=layer_idx,
            use_dara=use_dara,
            use_pai=use_pai,
            max_images=max_images,
            vis_token_amplifier=vis_token_amplifier,
        )

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, self_attn_weight
        return hidden_states

class My_Qwen3VLTextModel(Qwen3VLTextModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        use_dara: bool = False,
        use_pai: bool = False,
        max_images: int = 0,
        vis_token_amplifier: float = 0.0,
        enable_attn_shift: bool = True,
        attn_shift_multihead: bool = True,
        learnable_mu: bool = True,
    ):
        super().__init__(config)

        # # ✅ create approximator at model-level (like Code2 module-level learnables)
        # self.attn_approx = None
        # if enable_attn_shift:
        #     strat = ShiftStrategy.VECTOR_SHIFT
        #     if attn_shift_multihead:
        #         strat |= ShiftStrategy.MULTI_HEAD
        #     if learnable_mu:
        #         strat |= ShiftStrategy.LEARNABLE_SHIFT_SCALE

        #     self.attn_approx = Qwen3VLAttnApproximator(
        #         num_layers=config.num_hidden_layers,
        #         hidden_dim=config.hidden_size,
        #         num_heads=config.num_attention_heads,
        #         attn_strategy=strat,
        #     )
        #     self.attn_approx.enable_all()

        # build layers
        if use_dara:
            self.layers = nn.ModuleList(
                [
                    My_Qwen3VLTextDecoderLayer(
                        config,
                        layer_idx,
                        use_dara=True if layer_idx == 0 else False,
                        max_images=max_images if layer_idx == 0 else 0,
                        vis_token_amplifier=0.0,
                    )
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    My_Qwen3VLTextDecoderLayer(
                        config,
                        layer_idx,
                        use_pai=use_pai,
                        vis_token_amplifier=vis_token_amplifier,
                    )
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )

        # # ✅ wire attention layers to find this model (so they can use self.attn_approx)
        # for layer in self.layers:
        # #     layer.self_attn.parent_model = self
        # import weakref

        # # after building self.layers
        # for layer in self.layers:
        #     # IMPORTANT: write into __dict__ to avoid module registration
        #     layer.self_attn.__dict__["_parent_ref"] = weakref.ref(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # deepstack args
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

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
        all_hidden_states = () if output_hidden_states else None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
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
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if output_attentions:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                attentions=all_self_attns,
                hidden_states=all_hidden_states
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
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
            ).to(torch.bfloat16)

        # 3) init log_Z1_lin weights/bias explicitly
        if hasattr(self.attn_approx, "log_Z1_lin"):
            for m in self.attn_approx.log_Z1_lin:
                if isinstance(m, MultiheadLinear):
                    m.weight.data = (torch.randn_like(m.weight, dtype=torch.bfloat16, device=device) * init_std).to(torch.bfloat16)
                    m.bias.data = torch.zeros_like(m.bias, dtype=torch.bfloat16, device=device).to(torch.bfloat16)
                elif isinstance(m, nn.Linear):
                    m.weight.data = (torch.randn_like(m.weight, dtype=torch.bfloat16, device=device) * init_std).to(torch.bfloat16)
                    if m.bias is not None:
                        m.bias.data = torch.zeros_like(m.bias, dtype=torch.bfloat16, device=device).to(torch.bfloat16)

            # # optional: keep logZ1 in fp32 for stability
            # if keep_logZ1_fp32:
            #     self.attn_approx.log_Z1_lin = self.attn_approx.log_Z1_lin.float()

        self.attn_approx.enable_all()

        # ---- wire parent refs so attention can find attn_approx ----
        import weakref
        for layer in self.layers:
            layer.self_attn.__dict__["_parent_ref"] = weakref.ref(self)

        # sanity check: make sure no meta params remain in attn_approx
        bad_meta = [n for n, p in self.attn_approx.named_parameters() if getattr(p, "is_meta", False)]
        if bad_meta:
            raise RuntimeError(f"attn_approx still has meta params: {bad_meta[:10]}")


class My_Qwen3VLModel(Qwen3VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "My_Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config, use_dara=False, use_pai=False, use_icot=False, max_images=0, vis_token_amplifier=0.0,
                 enable_attn_shift=True, attn_shift_multihead=True, learnable_mu=True):
        super().__init__(config)
        self.language_model = My_Qwen3VLTextModel._from_config(
            config.text_config,
            use_dara=use_dara,
            use_pai=use_pai,
            max_images=max_images,
            vis_token_amplifier=vis_token_amplifier,
            enable_attn_shift=enable_attn_shift,
            attn_shift_multihead=attn_shift_multihead,
            learnable_mu=learnable_mu,
        )

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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.use_icot:
            # NOTE: your original logic was odd (tensor in list). keep as-is but safer:
            last_tok = input_ids[:, -1]
            judge = (last_tok == 382) | (last_tok == 624)
            if judge.any():
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
            attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

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
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

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
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states
            )
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            hidden_states=outputs.hidden_states
        )

class My_Qwen3VLForConditionalGeneration_mimic(Qwen3VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config, use_dara=False, use_pai=False, use_icot=False, max_images=0, vis_token_amplifier=0.0,
                 enable_attn_shift=True, attn_shift_multihead=True, learnable_mu=True):
        super().__init__(config)
        self.model = My_Qwen3VLModel(
            config,
            use_dara=use_dara,
            use_pai=use_pai,
            use_icot=use_icot,
            max_images=max_images,
            vis_token_amplifier=vis_token_amplifier,
            enable_attn_shift=enable_attn_shift,
            attn_shift_multihead=attn_shift_multihead,
            learnable_mu=learnable_mu,
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.use_pai = use_pai
        self.post_init()

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

        # NOTE: your original had `[...][0]` (batch0 only). Keep behavior, but safer if B>1:
        img_pos_ranges = [consecutive_ranges(ids, self.config.image_token_id) for ids in input_ids]
        if self.use_pai:
            new_ids, new_mask, lens = remove_visual_from_input_ids(input_ids, img_pos_ranges)

        # pass ranges for DARA / vis_token_amplifier
        kwargs["img_pos_ranges"] = img_pos_ranges[0] if len(img_pos_ranges) == 1 else img_pos_ranges[0]

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

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if self.use_pai:
            # WARNING: your original code used hidden_states again. Keeping your intent but fixing obvious bug:
            uncond_outputs = self.model(input_ids=new_ids, attention_mask=new_mask, use_cache=True)
            uncond_hidden = uncond_outputs[0]
            unconditional_logits = self.lm_head(uncond_hidden[:, slice_indices, :])

            cutoff = torch.log(torch.tensor(0.1, device=logits.device)) + logits.max(dim=-1, keepdim=True).values
            out = 0.5 * (logits - unconditional_logits) + unconditional_logits
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
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states
            )
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
            hidden_states=outputs.hidden_states
        )