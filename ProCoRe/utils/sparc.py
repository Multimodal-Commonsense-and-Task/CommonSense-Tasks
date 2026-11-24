"""SPARC (Selective Progressive Attention ReCalibration) for the eval_VLM stack.

This module is a model-agnostic adaptation of the original SPARC implementation
(https://github.com/.../SPARC, see /disk3/mjl/SPARC/attn_util.py) so that it can
be plugged into eval_VLM.py across the supported VLM families:

    - LLaVA (LLaMA-based)
    - InternVL3.5 (Qwen3 LM inside)
    - Qwen2-VL / Qwen3-VL
    - OneVision-1.5 (Qwen3-based)

SPARC needs eager attention (it mutates ``past_key_value.value_cache`` in place
between layers and reads softmax attention weights). Flash attention will not
work — install with ``--attn_impl eager``.
"""

from __future__ import annotations

import math
from functools import partial
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# Buffer that lives across layers and across generation steps
# ---------------------------------------------------------------------------
class SelectedIndexBuffer:
    """Mirror of SPARC/attn_util.SelectedIndexBuffer with a few extras for
    multi-family support (image_token_start is set per-sample by the caller
    instead of inferred from a hard-coded offset).
    """

    def __init__(self) -> None:
        self.indices1 = []           # selected token positions for current step
        self.indices2 = []           # selected token positions for previous step
        self.input_len = 0           # length of the prompt (excluding the final token)
        self.num_image_patches: Optional[int] = None
        self.image_token_start: Optional[int] = None  # absolute start index of image-token block
        # Absolute positions of every visual token. Unlike image_token_start /
        # num_image_patches, this supports multi-image ICL prompts whose visual
        # blocks are separated by demonstration text.
        self.image_token_indices: Optional[torch.Tensor] = None
        self.enabled: bool = True    # set False to bypass SPARC entirely

    # --- update helpers -----------------------------------------------------
    def update_indices1(self, indices: torch.Tensor, image_token_index: int) -> None:
        indices = indices.reshape(-1).long()
        if (
            isinstance(self.image_token_indices, torch.Tensor)
            and self.image_token_indices.numel() > 0
            and indices.numel() > 0
        ):
            image_positions = self.image_token_indices.to(indices.device)
            valid = indices < image_positions.numel()
            self.indices1 = image_positions[indices[valid]]
        else:
            self.indices1 = indices + image_token_index

    def update_indices2(self) -> None:
        self.indices2 = self.indices1

    def update_input_len(self, length: int) -> None:
        self.input_len = length

    def update_patch_num(self, num_image_patches: int) -> None:
        self.num_image_patches = num_image_patches

    def update_image_token_indices(self, indices: Optional[torch.Tensor]) -> None:
        self.image_token_indices = (
            indices.reshape(-1).long() if isinstance(indices, torch.Tensor) else None
        )

    def valid_image_token_indices(self, sequence_length: int, device) -> Optional[torch.Tensor]:
        if not isinstance(self.image_token_indices, torch.Tensor):
            return None
        indices = self.image_token_indices.to(device=device)
        indices = indices[(indices >= 0) & (indices < sequence_length)]
        return indices if indices.numel() > 0 else None

    def reset(self) -> None:
        self.indices1 = []
        self.indices2 = []
        self.input_len = 0
        self.num_image_patches = None
        self.image_token_start = None
        self.image_token_indices = None

    # --- in-place value-cache calibration ----------------------------------
    def calibrate(self, value_cache: torch.Tensor, alpha: float) -> None:
        if isinstance(self.indices2, torch.Tensor) and self.indices2.numel() > 0:
            value_cache[:, :, self.indices2] *= alpha
        elif isinstance(self.indices2, list) and len(self.indices2) > 0:
            value_cache[:, :, self.indices2] *= alpha


# ---------------------------------------------------------------------------
# Cache-API shim: transformers >=4.50 replaced ``cache.key_cache`` /
# ``cache.value_cache`` with ``cache.layers[i].keys`` / ``cache.layers[i].values``.
# ---------------------------------------------------------------------------
def _cache_has_layer(past_kv, layer_idx: int) -> bool:
    if past_kv is None:
        return False
    if hasattr(past_kv, "layers"):
        return (
            layer_idx < len(past_kv.layers)
            and getattr(past_kv.layers[layer_idx], "is_initialized", True)
            and past_kv.layers[layer_idx].keys is not None
        )
    if hasattr(past_kv, "key_cache"):
        return len(past_kv.key_cache) > layer_idx
    return False


def _cache_value_tensor(past_kv, layer_idx: int):
    """Return the value-cache tensor for ``layer_idx`` (or ``None``)."""
    if past_kv is None:
        return None
    if hasattr(past_kv, "layers"):
        if layer_idx < len(past_kv.layers):
            layer = past_kv.layers[layer_idx]
            vals = getattr(layer, "values", None)
            return vals
        return None
    if hasattr(past_kv, "value_cache") and len(past_kv.value_cache) > layer_idx:
        return past_kv.value_cache[layer_idx]
    return None


# ---------------------------------------------------------------------------
# Image-token locators
# ---------------------------------------------------------------------------
def locate_image_token_indices(
    model_name: str,
    input_ids: torch.Tensor,
    model=None,
    processor=None,
    tokenizer=None,
) -> Optional[torch.Tensor]:
    """Return all absolute visual-token positions in a prompt.

    InternVL and Idefics expose repeated visual-token IDs directly. Qwen-VL
    wraps each visual block in start/end markers, so collect every enclosed
    range. LLaVA expands its placeholder only inside the model and therefore
    continues to use the legacy span fallback in ``locate_image_span``.
    """
    name = model_name.lower()
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids

    if "intern" in name:
        img_ctx_id = getattr(model, "img_context_token_id", None)
        if img_ctx_id is None and tokenizer is not None:
            try:
                img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            except Exception:
                img_ctx_id = None
        if img_ctx_id is None:
            return None
        positions = (ids == img_ctx_id).nonzero(as_tuple=False).reshape(-1)
        return positions if positions.numel() > 0 else None

    if "idefics" in name:
        img_id = getattr(getattr(model, "config", None), "image_token_id", None)
        if img_id is None and processor is not None:
            tk = getattr(processor, "tokenizer", processor)
            try:
                img_id = tk.convert_tokens_to_ids("<image>")
            except Exception:
                img_id = None
        if img_id is None:
            return None
        positions = (ids == img_id).nonzero(as_tuple=False).reshape(-1)
        return positions if positions.numel() > 0 else None

    if "qwen" in name or "onevision" in name:
        vs_id = getattr(getattr(model, "config", None), "vision_start_token_id", None)
        ve_id = getattr(getattr(model, "config", None), "vision_end_token_id", None)
        if (vs_id is None or ve_id is None) and processor is not None:
            tk = getattr(processor, "tokenizer", processor)
            try:
                vs_id = tk.convert_tokens_to_ids("<|vision_start|>")
                ve_id = tk.convert_tokens_to_ids("<|vision_end|>")
            except Exception:
                return None
        starts = (ids == vs_id).nonzero(as_tuple=False).reshape(-1)
        ends = (ids == ve_id).nonzero(as_tuple=False).reshape(-1)
        ranges = []
        end_idx = 0
        for start in starts.tolist():
            while end_idx < ends.numel() and int(ends[end_idx]) <= start:
                end_idx += 1
            if end_idx >= ends.numel():
                break
            end = int(ends[end_idx])
            if end > start + 1:
                ranges.append(torch.arange(start + 1, end, device=ids.device))
            end_idx += 1
        return torch.cat(ranges) if ranges else None

    return None


def locate_image_span(
    model_name: str,
    input_ids: torch.Tensor,
    model=None,
    processor=None,
    tokenizer=None,
) -> Tuple[Optional[int], Optional[int]]:
    """Find a legacy contiguous image span inside ``input_ids``.

    Returns ``(start_index, num_image_patches)``. If no image tokens are
    detected the function returns ``(None, None)`` and the caller should
    skip SPARC for this sample. Multi-image-capable callers should prefer
    ``locate_image_token_indices``.
    """
    name = model_name.lower()
    if input_ids.dim() == 2:
        ids = input_ids[0]
    else:
        ids = input_ids

    # ---- LLaVA: IMAGE_TOKEN_INDEX == -200 placeholder ---------------------
    if "llava" in name and "onevision" not in name:
        # After ``prepare_inputs_labels_for_multimodal`` the image patches are
        # injected and the placeholder is replaced. We rely on the model to
        # still expose the placeholder location *before* the substitution,
        # which `model.generate` does internally. Practically we approximate
        # the start as the position of the first IMAGE_TOKEN_INDEX (-200) and
        # `num_image_patches` from the vision tower.
        try:
            image_token_index = (ids == -200).nonzero(as_tuple=False)
            if image_token_index.numel() == 0:
                return None, None
            start = int(image_token_index[0].item())
            # LLaVA expands one -200 token into N visual patches at runtime.
            num_patches = int(getattr(model.get_vision_tower(), "num_patches", 576))
            return start, num_patches
        except Exception:
            return None, None

    # ---- InternVL3.5: <IMG_CONTEXT> token positions -----------------------
    if "intern" in name:
        img_ctx_id = getattr(model, "img_context_token_id", None)
        if img_ctx_id is None and tokenizer is not None:
            try:
                img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            except Exception:
                img_ctx_id = None
        if img_ctx_id is None:
            return None, None
        positions = (ids == img_ctx_id).nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            return None, None
        # Preserve this legacy summary for compatibility. SPARC itself uses
        # locate_image_token_indices(), which retains discontiguous blocks.
        return int(positions[0].item()), int(positions.numel())

    # ---- Idefics3: contiguous <image> token id run -----------------------
    if "idefics" in name:
        img_id = getattr(getattr(model, "config", None), "image_token_id", None)
        if img_id is None and processor is not None:
            tk = getattr(processor, "tokenizer", processor)
            try:
                img_id = tk.convert_tokens_to_ids("<image>")
            except Exception:
                img_id = None
        if img_id is None:
            return None, None
        positions = (ids == img_id).nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            return None, None
        # Use the contiguous run that contains the LAST <image> token (the
        # query image in ICL setups).
        return int(positions[0].item()), int(positions.numel())

    # ---- Qwen2/3-VL & OneVision-1.5: <|vision_start|> ... <|vision_end|> --
    if "qwen" in name or "onevision" in name:
        vs_id = getattr(getattr(model, "config", None), "vision_start_token_id", None)
        ve_id = getattr(getattr(model, "config", None), "vision_end_token_id", None)
        if (vs_id is None or ve_id is None) and processor is not None:
            tk = getattr(processor, "tokenizer", processor)
            try:
                vs_id = tk.convert_tokens_to_ids("<|vision_start|>")
                ve_id = tk.convert_tokens_to_ids("<|vision_end|>")
            except Exception:
                return None, None
        vs = (ids == vs_id).nonzero(as_tuple=False)
        ve = (ids == ve_id).nonzero(as_tuple=False)
        if vs.numel() == 0 or ve.numel() == 0:
            return None, None
        start = int(vs[0].item()) + 1
        end = int(ve[0].item())
        return start, max(end - start, 0)

    return None, None


# ---------------------------------------------------------------------------
# LLaMA-style SPARC attention forward (verbatim from SPARC/attn_util.py with
# a small generalization of the layer-locator).
# ---------------------------------------------------------------------------
def _llama_sparc_forward(
    self,
    hidden_states: torch.Tensor,
    *args,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tau: float = 2.0,
    selected: bool = False,
    se_layers: Tuple[int, int] = (0, 31),
    indices_buffer: Optional[SelectedIndexBuffer] = None,
    return_present_pkv: bool = True,
    **kwargs,
):
    # Some custom LLaMA attention classes (e.g. /disk3/mjl/LMM-ICL/models/idefics3
    # ``My_LlamaAttention``) take ``position_embeddings`` as the 2nd positional
    # argument and ``past_key_value`` as singular. Accept both call shapes.
    if args and position_embeddings is None:
        position_embeddings = args[0]
        args = args[1:]
    if args and attention_mask is None:
        attention_mask = args[0]
        args = args[1:]
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # ``num_(_key_value)_heads`` may not be set on every attention class — infer
    # via ``-1`` when missing.
    num_q_heads = getattr(self, "num_heads", -1)
    num_kv_heads = getattr(self, "num_key_value_heads", -1)
    query_states = query_states.view(bsz, q_len, num_q_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if indices_buffer is not None and indices_buffer.enabled and self.layer_idx == 0:
        indices_buffer.update_indices2()

    if (
        indices_buffer is not None and indices_buffer.enabled
        and se_layers[0] <= self.layer_idx <= se_layers[1]
        and len(indices_buffer.indices2) > 0
    ):
        v_cache = _cache_value_tensor(past_key_value, self.layer_idx)
        if v_cache is not None:
            indices_buffer.calibrate(v_cache, alpha)

    gen_new_token = _cache_has_layer(past_key_value, self.layer_idx)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # ---- SPARC bookkeeping -------------------------------------------------
    if indices_buffer is not None and indices_buffer.enabled:
        if not gen_new_token and self.layer_idx == 0:
            if indices_buffer.num_image_patches is None:
                indices_buffer.update_patch_num(attn_weights.shape[-1] - indices_buffer.input_len)

        img_start = indices_buffer.image_token_start if indices_buffer.image_token_start is not None else 35
        image_positions = indices_buffer.valid_image_token_indices(
            attn_weights.shape[-1], attn_weights.device
        )
        n_patches = indices_buffer.num_image_patches or 0
        if image_positions is not None:
            image_attention = attn_weights[:, :, -1, :].index_select(
                -1, image_positions
            ).mean(dim=1)
        elif n_patches > 0 and img_start + n_patches <= attn_weights.shape[-1]:
            image_attention = attn_weights[:, :, -1, img_start : img_start + n_patches].mean(dim=1)
        else:
            image_attention = None

        if gen_new_token and selected and image_attention is not None and getattr(self, "image_attention", None) is not None:
            ratio = (image_attention - self.image_attention) / self.image_attention
            ratio = ratio.squeeze(dim=0)
            indices = (ratio >= tau).nonzero()
            indices_buffer.update_indices1(indices, img_start)

        if image_attention is not None:
            if not gen_new_token:
                self.image_attention = image_attention
            else:
                prev = getattr(self, "image_attention", image_attention)
                self.image_attention = (1 - beta) * image_attention + beta * prev

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    if return_present_pkv:
        return attn_output, attn_weights, past_key_value
    return attn_output, attn_weights


# Qwen2 attention is LLaMA-shaped (no q_norm/k_norm, returns 3-tuple) so it can
# share the LLaMA forward.
_qwen2_sparc_forward = _llama_sparc_forward


# ---------------------------------------------------------------------------
# Qwen3 / Qwen3-VL attention forward
#   - has ``self.q_norm`` / ``self.k_norm`` applied to Q/K
#   - takes ``position_embeddings`` as a positional/required argument
#   - signature: forward(hidden_states, position_embeddings, attention_mask=None,
#                        past_key_values=None, cache_position=None, **kwargs)
#   - returns (attn_output, attn_weights)  -- NOT 3 elements
#   - infers head count via ``view(..., -1, head_dim)`` (no ``num_key_value_heads``)
# ---------------------------------------------------------------------------
def _qwen3_sparc_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    tau: float = 2.0,
    selected: bool = False,
    se_layers: Tuple[int, int] = (0, 31),
    indices_buffer: Optional[SelectedIndexBuffer] = None,
    return_present_pkv: bool = False,
    **kwargs,
):
    # Some attention classes (OneVision-1.5) pass the cache as ``past_key_value``
    # (singular) rather than ``past_key_values``. Accept either.
    if past_key_values is None:
        past_key_values = kwargs.pop("past_key_value", None)
    else:
        kwargs.pop("past_key_value", None)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ---- SPARC step: snapshot indices for this generation step ------------
    if indices_buffer is not None and indices_buffer.enabled and self.layer_idx == 0:
        indices_buffer.update_indices2()

    # ---- SPARC step: calibrate past value cache before appending ----------
    has_indices2 = (
        (isinstance(indices_buffer.indices2, torch.Tensor) and indices_buffer.indices2.numel() > 0)
        or (isinstance(indices_buffer.indices2, list) and len(indices_buffer.indices2) > 0)
    ) if indices_buffer is not None else False

    if (
        indices_buffer is not None and indices_buffer.enabled
        and se_layers[0] <= self.layer_idx <= se_layers[1]
        and has_indices2
    ):
        v_cache = _cache_value_tensor(past_key_values, self.layer_idx)
        if v_cache is not None:
            indices_buffer.calibrate(v_cache, alpha)

    gen_new_token = _cache_has_layer(past_key_values, self.layer_idx)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # ---- eager attention (manually, so we get attn_weights) ---------------
    key_states_g = repeat_kv(key_states, self.num_key_value_groups)
    value_states_g = repeat_kv(value_states, self.num_key_value_groups)

    scaling = getattr(self, "scaling", 1.0 / math.sqrt(self.head_dim))
    attn_weights = torch.matmul(query_states, key_states_g.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states_g.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if self.training and getattr(self, "attention_dropout", 0.0) > 0.0:
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

    # ---- SPARC bookkeeping on attn_weights --------------------------------
    if indices_buffer is not None and indices_buffer.enabled:
        if not gen_new_token and self.layer_idx == 0 and indices_buffer.num_image_patches is None:
            indices_buffer.update_patch_num(attn_weights.shape[-1] - indices_buffer.input_len)

        img_start = indices_buffer.image_token_start
        image_positions = indices_buffer.valid_image_token_indices(
            attn_weights.shape[-1], attn_weights.device
        )
        n_patches = indices_buffer.num_image_patches or 0
        if image_positions is not None:
            image_attention = attn_weights[:, :, -1, :].index_select(
                -1, image_positions
            ).mean(dim=1)
        elif img_start is not None and n_patches > 0 and img_start + n_patches <= attn_weights.shape[-1]:
            image_attention = attn_weights[:, :, -1, img_start : img_start + n_patches].mean(dim=1)
        else:
            image_attention = None

        if (
            gen_new_token and selected and image_attention is not None
            and getattr(self, "image_attention", None) is not None
        ):
            ratio = (image_attention - self.image_attention) / self.image_attention
            ratio = ratio.squeeze(dim=0)
            indices = (ratio >= tau).nonzero()
            indices_buffer.update_indices1(indices, img_start)

        if image_attention is not None:
            if not gen_new_token:
                self.image_attention = image_attention
            else:
                prev = getattr(self, "image_attention", image_attention)
                self.image_attention = (1 - beta) * image_attention + beta * prev

    attn_output = torch.matmul(attn_weights, value_states_g)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    if not kwargs.get("output_attentions", False):
        attn_weights = None
    if return_present_pkv:
        # OneVision-1.5 / LLaMA-shaped decoder layers expect 3-tuple.
        return attn_output, attn_weights, past_key_values
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Attach SPARC to a model
# ---------------------------------------------------------------------------
def _resolve_layers(model, model_name: str):
    """Return ``(layers, family)`` for the decoder layers SPARC should patch.

    ``family`` is determined dynamically from the attention class so we don't
    need to maintain a hard-coded name-prefix table — Qwen3-style attention
    (with ``q_norm``/``k_norm``) gets the Qwen3 forward, everything else
    gets the LLaMA forward.
    """
    name = model_name.lower()

    # ---- locate the decoder ModuleList -----------------------------------
    layers = None
    if "intern" in name:
        lm = getattr(model, "language_model", None)
        if lm is None:
            lm = getattr(getattr(model, "base_model", model), "language_model", None)
        inner = getattr(lm, "model", None) or lm
        layers = inner.layers
    elif "idefics" in name:
        # Idefics3ForConditionalGeneration -> .model -> .text_model -> .layers
        try:
            layers = model.model.text_model.layers
        except AttributeError:
            layers = None
    elif "qwen" in name or "onevision" in name:
        if hasattr(model, "language_model"):
            inner = getattr(model.language_model, "model", model.language_model)
            layers = inner.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
    elif "llava" in name:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "language_model"):
            inner = getattr(model.language_model, "model", model.language_model)
            layers = inner.layers

    if layers is None:
        # Fallback: probe common attribute paths
        for fn in (
            lambda m: m.model.text_model.layers,                # Idefics3
            lambda m: m.model.layers,                            # LLaVA / Qwen-only
            lambda m: m.language_model.model.layers,             # Qwen-VL / OneVision
            lambda m: m.language_model.layers,
            lambda m: m.base_model.model.model.layers,           # PEFT-wrapped
            lambda m: m.base_model.model.model.text_model.layers,
        ):
            try:
                layers = fn(model)
                break
            except AttributeError:
                continue
        if layers is None:
            raise RuntimeError(f"Could not resolve decoder layers for model: {model_name}")

    # ---- pick family by inspecting the attention module ------------------
    import inspect as _inspect
    import re as _re

    sample_attn = layers[0].self_attn
    has_qnorm = hasattr(sample_attn, "q_norm") and hasattr(sample_attn, "k_norm")

    # Detect the return arity by examining the *decoder layer's* unpacking
    # pattern around its `self.self_attn(...)` call — that's the source of
    # truth for what the caller will accept. Falls back to scanning the
    # attention class's own ``return`` statements if that fails.
    returns_3tuple = None  # tri-state: None=unknown, True/False=detected

    sample_layer = layers[0]
    try:
        layer_src = _inspect.getsource(type(sample_layer).forward)
    except (TypeError, OSError):
        layer_src = ""
    if layer_src:
        # Find every assignment of the form ``a, b = self.self_attn(`` or
        # ``a, b, c = self.self_attn(``. The number of LHS names tells us
        # the expected arity.
        m = _re.search(r"([\w\s,]+?)\s*=\s*self\.self_attn\s*\(", layer_src)
        if m:
            lhs = m.group(1)
            n_names = len([t for t in lhs.split(",") if t.strip()])
            if n_names == 3:
                returns_3tuple = True
            elif n_names == 2:
                returns_3tuple = False

    if returns_3tuple is None:
        # Fallback: count commas on `return ...` lines of the attn class.
        try:
            attn_src = _inspect.getsource(type(sample_attn).forward)
        except (TypeError, OSError):
            attn_src = ""
        for line in attn_src.splitlines():
            s = line.strip()
            if s.startswith("return "):
                payload = s[len("return "):]
                commas = sum(1 for ch in payload if ch == ",")
                returns_3tuple = (commas >= 2)
                break
    if returns_3tuple is None:
        returns_3tuple = True  # safe default for older transformers

    if has_qnorm:
        family = "qwen3_3tuple" if returns_3tuple else "qwen3"
    else:
        family = "llama_3tuple" if returns_3tuple else "llama_2tuple"
    return layers, family


def apply_sparc(
    model,
    *,
    model_name: str,
    alpha: float = 1.1,
    beta: float = 0.1,
    tau: float = 1.5,
    selected_layer: int = 20,
    se_layers: Tuple[int, int] = (0, 31),
) -> SelectedIndexBuffer:
    """Patch every decoder layer's ``self_attn.forward`` with SPARC."""
    buffer = SelectedIndexBuffer()
    layers, family = _resolve_layers(model, model_name)
    if family == "qwen3":
        fwd_fn = _qwen3_sparc_forward
        extra = {"return_present_pkv": False}
    elif family == "qwen3_3tuple":
        fwd_fn = _qwen3_sparc_forward
        extra = {"return_present_pkv": True}
    elif family == "llama_2tuple":
        fwd_fn = _llama_sparc_forward
        extra = {"return_present_pkv": False}
    else:  # "llama_3tuple" or fallback "llama"
        fwd_fn = _llama_sparc_forward
        extra = {"return_present_pkv": True}

    for i, layer in enumerate(layers):
        selected = (i == selected_layer)
        wrapped = partial(
            fwd_fn,
            alpha=alpha,
            beta=beta,
            tau=tau,
            selected=selected,
            se_layers=se_layers,
            indices_buffer=buffer,
            **extra,
        )
        layer.self_attn.forward = MethodType(wrapped, layer.self_attn)

    return buffer


# ---------------------------------------------------------------------------
# Wrap model.generate / model.chat to set up the buffer per sample
# ---------------------------------------------------------------------------
def wrap_generate_with_sparc(
    model,
    buffer: SelectedIndexBuffer,
    model_name: str,
    processor=None,
    tokenizer=None,
) -> None:
    """Monkey-patch ``model.generate`` (and ``model.chat`` for InternVL) so the
    SPARC buffer is populated/reset per sample without touching eval_VLM.py's
    per-family branches."""

    orig_generate = model.generate

    def _setup(input_ids):
        try:
            image_indices = locate_image_token_indices(
                model_name, input_ids, model, processor, tokenizer
            )
        except Exception:
            image_indices = None
        try:
            start, count = locate_image_span(model_name, input_ids, model, processor, tokenizer)
        except Exception:
            start, count = None, None
        buffer.update_input_len(input_ids.shape[-1] - 1)
        buffer.image_token_start = start
        buffer.update_patch_num(count if count else None)
        buffer.update_image_token_indices(image_indices)

    def sparc_generate(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if isinstance(ids, torch.Tensor):
            _setup(ids)
        try:
            out = orig_generate(*args, **kwargs)
        finally:
            buffer.reset()
        return out

    model.generate = sparc_generate

    # InternVL exposes a chat() helper that internally calls generate but the
    # caller passes a question string, not input_ids. Wrap it so the per-sample
    # reset still happens around the whole chat call.
    if hasattr(model, "chat"):
        orig_chat = model.chat

        def sparc_chat(tokenizer_, pixel_values, question, generation_config,
                       num_patches_list=None, history=None, return_history=False, **kw):
            try:
                return orig_chat(
                    tokenizer_, pixel_values, question, generation_config,
                    num_patches_list=num_patches_list, history=history,
                    return_history=return_history, **kw,
                )
            finally:
                buffer.reset()

        model.chat = sparc_chat
