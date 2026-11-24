"""Core VGPO reweighting operations.

This module contains the tensor-only parts of Visually-Guided Policy
Optimization so they can be tested independently of TRL and the VLM runtime.
The equations follow the reference implementation in ``../VGPO``:

* visual attention compensation (Eq. 5-6),
* intra-trajectory advantage reweighting (Eq. 7-8), and
* inter-trajectory advantage reweighting (Eq. 9-11).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_visual_focus_scores(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Compute cosine similarity between completion tokens and visual prototypes.

    The visual prototype for a sequence is the mean final hidden state of all
    image tokens in its prompt. Returned scores are detached because VGPO uses
    them as credit-assignment weights, not as an auxiliary differentiable loss.
    """
    completion_length = completion_mask.size(1)
    if last_hidden_state.shape[:2] != input_ids.shape:
        raise ValueError(
            "VGPO requires hidden states aligned with input_ids, but received "
            f"{tuple(last_hidden_state.shape[:2])} and {tuple(input_ids.shape)}."
        )
    if completion_length > input_ids.size(1):
        raise ValueError(
            f"Completion length {completion_length} exceeds sequence length "
            f"{input_ids.size(1)}."
        )

    prompt_length = input_ids.size(1) - completion_length
    prompt_positions = torch.arange(input_ids.size(1), device=input_ids.device)
    prompt_positions = prompt_positions.unsqueeze(0) < prompt_length
    visual_mask = (
        input_ids.eq(image_token_id)
        & attention_mask.bool()
        & prompt_positions
    )

    hidden = last_hidden_state.float()
    visual_count = visual_mask.sum(dim=1, keepdim=True)
    visual_prototype = (
        hidden * visual_mask.unsqueeze(-1)
    ).sum(dim=1) / visual_count.clamp(min=1).to(hidden.dtype)

    completion_hidden = hidden[:, -completion_length:, :]
    focus_scores = F.cosine_similarity(
        completion_hidden,
        visual_prototype.unsqueeze(1),
        dim=-1,
        eps=1e-8,
    )

    valid_completion = completion_mask.bool()
    has_visual_tokens = visual_count.squeeze(1).gt(0).unsqueeze(1)
    focus_scores = focus_scores.masked_fill(
        ~(valid_completion & has_visual_tokens), 0.0
    )
    return focus_scores.detach()


def _global_minmax(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    normalized = torch.zeros_like(values, dtype=torch.float32)
    valid = mask.bool()
    if not valid.any():
        return normalized

    valid_values = values.float()[valid]
    value_min = valid_values.min()
    value_range = valid_values.max() - value_min
    if value_range <= eps:
        return normalized

    normalized[valid] = (valid_values - value_min) / value_range
    return normalized


def _per_sequence_minmax(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    normalized = torch.zeros_like(values, dtype=torch.float32)
    for row in range(values.size(0)):
        valid = mask[row].bool()
        if not valid.any():
            continue
        valid_values = values[row, valid].float()
        value_min = valid_values.min()
        value_range = valid_values.max() - value_min
        if value_range > eps:
            normalized[row, valid] = (valid_values - value_min) / value_range
    return normalized


def compute_intra_trajectory_weights(
    visual_focus_scores: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    use_visual_compensation: bool = True,
    use_gated_visual_compensation: bool = True,
    visual_compensation_strength: float = 0.3,
    gated_visual_compensation_start_ratio: float = 0.5,
    visual_attention_threshold: float = 0.2,
    clamp_min: float = 0.1,
    clamp_max: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized focus scores and Eq. 8 token advantage factors."""
    if visual_focus_scores.shape != completion_mask.shape:
        raise ValueError(
            "visual_focus_scores and completion_mask must have the same shape, "
            f"got {tuple(visual_focus_scores.shape)} and "
            f"{tuple(completion_mask.shape)}."
        )
    if not 0.0 <= gated_visual_compensation_start_ratio <= 1.0:
        raise ValueError(
            "gated_visual_compensation_start_ratio must be in [0, 1]."
        )
    if not 0.0 <= visual_attention_threshold <= 1.0:
        raise ValueError("visual_attention_threshold must be in [0, 1].")
    if visual_compensation_strength < 0.0:
        raise ValueError("visual_compensation_strength must be non-negative.")

    valid_mask = completion_mask.bool()
    focus = _global_minmax(visual_focus_scores, valid_mask)

    if use_visual_compensation:
        compensated = focus.clone()
        for row in range(focus.size(0)):
            valid_indices = valid_mask[row].nonzero(as_tuple=True)[0]
            token_count = valid_indices.numel()
            if token_count == 0:
                continue

            if token_count == 1:
                relative_positions = torch.zeros(
                    1, device=focus.device, dtype=torch.float32
                )
            else:
                relative_positions = torch.linspace(
                    0.0,
                    1.0,
                    token_count,
                    device=focus.device,
                    dtype=torch.float32,
                )

            if use_gated_visual_compensation:
                gate = torch.zeros(
                    token_count, device=focus.device, dtype=torch.float32
                )
                start = int(
                    token_count * gated_visual_compensation_start_ratio
                )
                if start < token_count:
                    tail_weights = focus[row, valid_indices[start:]]
                    top_count = max(
                        1, int(tail_weights.numel() * visual_attention_threshold)
                    )
                    threshold = torch.topk(
                        tail_weights, k=top_count, largest=True
                    ).values[-1]
                    gate[start:] = tail_weights.ge(threshold).float()
            else:
                gate = torch.ones(
                    token_count, device=focus.device, dtype=torch.float32
                )

            compensation = (
                visual_compensation_strength * relative_positions * gate
            )
            compensated[row, valid_indices] = (
                focus[row, valid_indices] * (1.0 + compensation)
            )
        focus = compensated

    normalized_focus = _per_sequence_minmax(focus, valid_mask)
    token_weights = torch.ones_like(normalized_focus)
    for row in range(normalized_focus.size(0)):
        valid = valid_mask[row]
        if not valid.any():
            continue
        valid_weights = normalized_focus[row, valid]
        token_weights[row, valid] = 1.0 + (
            valid_weights - valid_weights.mean()
        )

    token_weights = token_weights.clamp(min=clamp_min, max=clamp_max)
    token_weights = token_weights.masked_fill(~valid_mask, 1.0)
    return normalized_focus, token_weights


def compute_visual_accumulation(
    normalized_focus: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the trajectory visual accumulation phi_i from Eq. 9."""
    return (
        normalized_focus.float() * completion_mask.to(torch.float32)
    ).sum(dim=1)


def compute_inter_trajectory_weights(
    visual_accumulation: torch.Tensor,
    num_generations: int,
    *,
    clamp_min: float = 0.9,
    clamp_max: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean-centered rollout-group factors from Eq. 10-11."""
    if visual_accumulation.ndim != 1:
        raise ValueError("visual_accumulation must be a one-dimensional tensor.")
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for VGPO.")
    if visual_accumulation.numel() % num_generations != 0:
        raise ValueError(
            "The rollout batch must contain complete generation groups: "
            f"{visual_accumulation.numel()} trajectories cannot be grouped by "
            f"{num_generations}."
        )

    grouped = visual_accumulation.float().view(-1, num_generations)
    group_min = grouped.min(dim=1, keepdim=True).values
    group_range = grouped.max(dim=1, keepdim=True).values - group_min
    normalized = torch.where(
        group_range > eps,
        (grouped - group_min) / group_range.clamp(min=eps),
        torch.zeros_like(grouped),
    )
    factors = 1.0 + (normalized - normalized.mean(dim=1, keepdim=True))
    return factors.clamp(min=clamp_min, max=clamp_max).reshape(-1)
