"""Perception-Exploration Policy Optimization (PEPO) utilities.

The equations in this module are ported from
``/home/ubuntu/mjlee/PEPO/src/pepo/train/grpo_trainer.py`` without depending
on PEPO's ms-swift training stack.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


VISION_SCORE_TYPES = {"cosine", "l2", "l1", "dot"}


def validate_pepo_config(
    *,
    gate_alpha: float,
    gate_temperature: float,
    vision_score_type: str,
) -> None:
    if gate_alpha < 0:
        raise ValueError("PEPO gate_alpha must be non-negative.")
    if gate_temperature <= 0:
        raise ValueError("PEPO gate_temperature must be positive.")
    if vision_score_type not in VISION_SCORE_TYPES:
        raise ValueError(
            "PEPO vision_score_type must be one of "
            f"{sorted(VISION_SCORE_TYPES)}, got {vision_score_type!r}."
        )


def _image_token_ranges(
    input_ids: torch.Tensor,
    image_token_id: int,
) -> list[torch.Tensor]:
    ranges = []
    for ids in input_ids:
        indices = ids.eq(image_token_id).nonzero(as_tuple=True)[0]
        # Preserve the official PEPO fallback for processors that replace image
        # placeholders before this forward pass.
        if indices.numel() == 0:
            indices = torch.arange(1, ids.numel(), device=ids.device)
        ranges.append(indices)
    return ranges


def compute_vision_scores(
    hidden_states: Iterable[torch.Tensor],
    input_ids: torch.Tensor,
    logits_to_keep: int,
    image_token_id: int,
    score_type: str = "cosine",
) -> torch.Tensor:
    """Average response-to-vision similarity over all model layers."""
    if score_type not in VISION_SCORE_TYPES:
        raise ValueError(f"Unknown PEPO vision score type: {score_type!r}.")
    hidden_states = tuple(hidden_states)
    if not hidden_states:
        raise ValueError("PEPO requires model hidden states.")

    token_ranges = _image_token_ranges(input_ids, image_token_id)
    first_hidden = hidden_states[0]
    batch_size = input_ids.size(0)
    accumulation_dtype = (
        torch.float32
        if first_hidden.is_cuda and first_hidden.dtype == torch.bfloat16
        else first_hidden.dtype
    )
    scores = torch.zeros(
        (batch_size, logits_to_keep),
        device=first_hidden.device,
        dtype=accumulation_dtype,
    )

    for layer_hidden in hidden_states:
        for row_index, vision_indices in enumerate(token_ranges):
            if vision_indices.numel() == 0:
                continue
            vision_hidden = layer_hidden[row_index].index_select(
                0, vision_indices
            )
            response_hidden = layer_hidden[row_index, -logits_to_keep:]

            if score_type == "cosine":
                response_hidden = F.normalize(response_hidden, dim=-1)
                vision_hidden = F.normalize(vision_hidden, dim=-1)
                token_scores = (
                    response_hidden @ vision_hidden.transpose(0, 1)
                ).mean(dim=1)
            elif score_type == "dot":
                token_scores = (
                    response_hidden @ vision_hidden.transpose(0, 1)
                ).mean(dim=1)
            elif score_type == "l2":
                response_sq = response_hidden.square().sum(
                    dim=1, keepdim=True
                )
                vision_sq = vision_hidden.square().sum(
                    dim=1, keepdim=True
                ).transpose(0, 1)
                squared_distance = (
                    response_sq
                    + vision_sq
                    - 2.0
                    * (response_hidden @ vision_hidden.transpose(0, 1))
                ).clamp_min(0.0)
                token_scores = -torch.sqrt(
                    squared_distance + 1e-6
                ).mean(dim=1)
            else:
                if (
                    response_hidden.is_cuda
                    and response_hidden.dtype == torch.bfloat16
                ):
                    response_hidden = response_hidden.float()
                    vision_hidden = vision_hidden.float()
                token_scores = -torch.cdist(
                    response_hidden, vision_hidden, p=1
                ).mean(dim=1)

            scores[row_index] += token_scores.to(accumulation_dtype)

    scores /= max(1, len(hidden_states))
    return scores.to(first_hidden.dtype)


def compute_pepo_token_weights(
    vision_scores: torch.Tensor,
    entropies: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    gate_alpha: float = 0.05,
    gate_temperature: float = 1.8,
) -> torch.Tensor:
    """Fuse visual grounding and entropy into mean-one token weights."""
    validate_pepo_config(
        gate_alpha=gate_alpha,
        gate_temperature=gate_temperature,
        vision_score_type="cosine",
    )
    if vision_scores.shape != entropies.shape:
        raise ValueError(
            "PEPO vision-score/entropy shape mismatch: "
            f"{tuple(vision_scores.shape)} vs {tuple(entropies.shape)}."
        )
    if vision_scores.shape != completion_mask.shape:
        raise ValueError(
            "PEPO score/mask shape mismatch: "
            f"{tuple(vision_scores.shape)} vs {tuple(completion_mask.shape)}."
        )

    mask = completion_mask.bool()
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    epsilon = 1e-6

    vision_min = vision_scores.masked_fill(
        ~mask, float("inf")
    ).min(dim=1, keepdim=True).values
    vision_max = vision_scores.masked_fill(
        ~mask, float("-inf")
    ).max(dim=1, keepdim=True).values
    normalized_vision = (vision_scores - vision_min) / (
        vision_max - vision_min
    ).clamp(min=epsilon)
    normalized_vision = torch.where(
        mask, normalized_vision, torch.zeros_like(normalized_vision)
    )

    entropy_min = entropies.masked_fill(
        ~mask, float("inf")
    ).min(dim=1, keepdim=True).values
    entropy_max = entropies.masked_fill(
        ~mask, float("-inf")
    ).max(dim=1, keepdim=True).values
    normalized_entropy = (entropies - entropy_min) / (
        entropy_max - entropy_min
    ).clamp(min=epsilon)
    normalized_entropy = torch.where(
        mask, normalized_entropy, torch.zeros_like(normalized_entropy)
    )

    mixed = normalized_vision + normalized_entropy
    mixed_mean = (mixed * mask).sum(dim=1, keepdim=True) / valid_counts
    gate = 1.0 + gate_alpha * torch.tanh(mixed - mixed_mean)

    gated_logits = vision_scores * gate / gate_temperature
    masked_logits = gated_logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(masked_logits, dim=1)
    weights = weights * valid_counts.to(weights.dtype)
    return torch.where(mask, weights, torch.zeros_like(weights))


def anneal_pepo_token_weights(
    token_weights: torch.Tensor,
    *,
    global_step: int,
    max_steps: int,
) -> torch.Tensor:
    """Linearly introduce PEPO weights over the course of training."""
    progress = min(1.0, max(0.0, global_step / max(1, max_steps)))
    return 1.0 - progress + progress * token_weights
