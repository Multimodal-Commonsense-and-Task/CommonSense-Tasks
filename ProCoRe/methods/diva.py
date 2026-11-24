"""DIVA-GRPO advantage utilities and structured rollout sampling.

This module ports the algorithmic pieces used by DIVA-GRPO to LMM-ICL's TRL
training stack.  It deliberately has no dependency on verl so that it can be
used by the existing DeepSpeed/GRPO entrypoint.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable, Optional, Sized

import torch
from torch.utils.data import Sampler


DEFAULT_SCORE_RANGES = (
    (0.0, 0.05),
    (0.05, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 1.0),
)
DEFAULT_DIFFICULTY_CHANGES = (2, 1, 0, -1, -2)


def parse_score_ranges(value: Any) -> list[tuple[float, float]]:
    """Parse JSON or ``low:high,low:high`` score-range notation."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(DEFAULT_SCORE_RANGES)
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = [part.split(":", 1) for part in stripped.split(",")]

    ranges = [(float(low), float(high)) for low, high in value]
    if not ranges:
        raise ValueError("DIVA score ranges cannot be empty.")
    if any(low > high for low, high in ranges):
        raise ValueError(f"Invalid DIVA score ranges: {ranges}")
    return ranges


def parse_difficulty_changes(value: Any) -> list[int]:
    """Parse JSON or comma-separated difficulty deltas."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(DEFAULT_DIFFICULTY_CHANGES)
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = stripped.split(",")
    changes = [int(change) for change in value]
    if not changes:
        raise ValueError("DIVA difficulty changes cannot be empty.")
    return changes


def validate_difficulty_schedule(
    score_ranges: list[tuple[float, float]],
    difficulty_changes: list[int],
) -> None:
    if len(score_ranges) != len(difficulty_changes):
        raise ValueError(
            "DIVA score ranges and difficulty changes must have the same "
            f"length, got {len(score_ranges)} and {len(difficulty_changes)}."
        )


def calculate_new_difficulty(
    average_score: float,
    old_difficulty: int,
    score_ranges: list[tuple[float, float]],
    difficulty_changes: list[int],
    min_difficulty: int,
    max_difficulty: int,
) -> int:
    """Apply DIVA's reward-conditioned difficulty update."""
    new_difficulty = old_difficulty
    for (low, high), change in zip(score_ranges, difficulty_changes):
        if low <= average_score <= high:
            new_difficulty += change
            break
    return max(min_difficulty, min(new_difficulty, max_difficulty))


def _grouped_advantages(
    rewards: torch.Tensor,
    group_ids: list[str],
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if rewards.ndim != 1:
        raise ValueError(f"Expected scalar rewards, got shape {tuple(rewards.shape)}.")
    if rewards.numel() != len(group_ids):
        raise ValueError(
            "Reward/group metadata length mismatch: "
            f"{rewards.numel()} rewards and {len(group_ids)} IDs."
        )

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped_indices[str(group_id)].append(index)

    advantages = torch.empty_like(rewards)
    for group_id, indices in grouped_indices.items():
        if len(indices) < 2:
            raise ValueError(
                f"DIVA-GRPO group {group_id!r} has only one rollout; "
                "set --grpo_num_gen to at least 2."
            )
        index_tensor = torch.tensor(indices, device=rewards.device)
        group_rewards = rewards[index_tensor]
        std = group_rewards.std()
        advantages[index_tensor] = (
            group_rewards - group_rewards.mean()
        ) / (std + epsilon)
    return advantages


def _zscore(values: torch.Tensor) -> torch.Tensor:
    std = values.std()
    if not torch.isfinite(std) or std <= 0:
        std = torch.ones((), device=values.device, dtype=values.dtype)
    return (values - values.mean()) / std


def _difficulty_weight(
    advantages: torch.Tensor,
    difficulty_differences: torch.Tensor,
    k: float,
) -> torch.Tensor:
    signs = torch.sign(advantages)
    multipliers = torch.exp(k * difficulty_differences * signs)
    return advantages * multipliers


def _reward_range_rescale(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: list[str],
    threshold: float,
    scale_factor: float,
) -> torch.Tensor:
    grouped_rewards: dict[str, list[float]] = defaultdict(list)
    for group_id, reward in zip(group_ids, rewards):
        grouped_rewards[str(group_id)].append(float(reward.item()))
    low_reward_groups = {
        group_id
        for group_id, group_rewards in grouped_rewards.items()
        if all(reward < threshold for reward in group_rewards)
    }
    mask = torch.tensor(
        [str(group_id) in low_reward_groups for group_id in group_ids],
        device=advantages.device,
        dtype=torch.bool,
    )
    return torch.where(mask, advantages * scale_factor, advantages)


def compute_diva_advantages(
    rewards: torch.Tensor,
    local_ids: list[str],
    global_ids: list[str],
    difficulties: Iterable[float],
    *,
    weight_mode: str,
    weighted_advantage_k: float,
    rrb_threshold: float,
    rrb_scale_factor: float,
    local_weight: float = 1.0,
    global_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DIVA's joint local/global advantages.

    ``local_ids`` identify a particular difficulty variant, while
    ``global_ids`` identify every variant derived from the same source sample.
    """
    tokens = {token.upper() for token in weight_mode.split("_") if token}
    local_advantages = _grouped_advantages(rewards, local_ids)
    global_advantages = _grouped_advantages(rewards, global_ids)

    difficulty_tensor = torch.as_tensor(
        list(difficulties), device=rewards.device, dtype=rewards.dtype
    )
    if difficulty_tensor.numel() != rewards.numel():
        raise ValueError(
            "Reward/difficulty metadata length mismatch: "
            f"{rewards.numel()} rewards and {difficulty_tensor.numel()} difficulties."
        )

    grouped_difficulties: dict[str, list[float]] = defaultdict(list)
    for group_id, difficulty in zip(global_ids, difficulty_tensor.tolist()):
        grouped_difficulties[str(group_id)].append(difficulty)
    group_means = {
        group_id: sum(values) / len(values)
        for group_id, values in grouped_difficulties.items()
    }
    difficulty_differences = torch.tensor(
        [
            difficulty - group_means[str(group_id)]
            for group_id, difficulty in zip(global_ids, difficulty_tensor.tolist())
        ],
        device=rewards.device,
        dtype=rewards.dtype,
    )

    if "ABSWBN" in tokens or "ABSWAN" in tokens:
        difficulty_differences = difficulty_tensor - difficulty_tensor.mean()

    if "WBN" in tokens or "ABSWBN" in tokens:
        global_advantages = _difficulty_weight(
            global_advantages, difficulty_differences, weighted_advantage_k
        )

    if "RMSNORM" in tokens:
        local_rms = local_advantages.square().mean().sqrt().clamp_min(1e-6)
        global_rms = global_advantages.square().mean().sqrt().clamp_min(1e-6)
        local_advantages = local_advantages / local_rms
        global_advantages = global_advantages / global_rms
    elif "MINMAXNORM" in tokens:
        local_scale = local_advantages.abs().max().clamp_min(1e-6)
        global_scale = global_advantages.abs().max().clamp_min(1e-6)
        local_advantages = local_advantages / local_scale
        global_advantages = global_advantages / global_scale
    elif "ZSCORENORM" in tokens:
        local_advantages = _zscore(local_advantages)
        global_advantages = _zscore(global_advantages)

    if "WAN" in tokens or "ABSWAN" in tokens:
        global_advantages = _difficulty_weight(
            global_advantages, difficulty_differences, weighted_advantage_k
        )

    if "RRBLOCAL" in tokens:
        local_advantages = _reward_range_rescale(
            local_advantages,
            rewards,
            local_ids,
            rrb_threshold,
            rrb_scale_factor,
        )
    if "RRBGLOBAL" in tokens:
        global_advantages = _reward_range_rescale(
            global_advantages,
            rewards,
            global_ids,
            rrb_threshold,
            rrb_scale_factor,
        )

    combined = local_weight * local_advantages + global_weight * global_advantages
    diagnostics = {
        "local": local_advantages,
        "global": global_advantages,
        "difficulty_difference": difficulty_differences,
    }
    return combined, diagnostics


def compute_difficulty_updates(
    rewards: torch.Tensor,
    source_indices: list[int],
    base_difficulties: list[int],
    score_ranges: list[tuple[float, float]],
    difficulty_changes: list[int],
    min_difficulty: int,
    max_difficulty: int,
) -> dict[int, int]:
    """Compute one dynamic difficulty update per source sample."""
    grouped_rewards: dict[int, list[float]] = defaultdict(list)
    old_difficulties: dict[int, int] = {}
    for source_index, base_difficulty, reward in zip(
        source_indices, base_difficulties, rewards
    ):
        source_index = int(source_index)
        grouped_rewards[source_index].append(float(reward.item()))
        old_difficulties[source_index] = int(base_difficulty)

    updates = {}
    for source_index, values in grouped_rewards.items():
        updates[source_index] = calculate_new_difficulty(
            sum(values) / len(values),
            old_difficulties[source_index],
            score_ranges,
            difficulty_changes,
            min_difficulty,
            max_difficulty,
        )
    return updates


class DIVARepeatSampler(Sampler[int]):
    """Keep every difficulty variant of a source in one GRPO rollout batch."""

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.seed = seed

        self.variants_per_source = int(
            getattr(data_source, "variants_per_source", 0)
        )
        self.base_size = int(getattr(data_source, "base_size", 0))
        if self.variants_per_source < 2 or self.base_size < 1:
            raise ValueError(
                "DIVARepeatSampler requires a DIVA adaptive dataset with at "
                "least one generated variant."
            )
        if batch_size % self.variants_per_source != 0:
            raise ValueError(
                "The number of unique prompts in a generation batch "
                f"({batch_size}) must be divisible by DIVA variants per source "
                f"({self.variants_per_source}). Adjust gradient accumulation, "
                "generation batch size, or DIVA_VARIANT_NUM."
            )
        self.sources_per_batch = batch_size // self.variants_per_source

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            source_indices = torch.randperm(
                self.base_size, generator=self.generator
            ).tolist()
        else:
            source_indices = list(range(self.base_size))

        chunks = [
            source_indices[index : index + self.sources_per_batch]
            for index in range(0, len(source_indices), self.sources_per_batch)
        ]
        chunks = [
            chunk for chunk in chunks if len(chunk) == self.sources_per_batch
        ]

        for chunk in chunks:
            expanded_indices = [
                source_index * self.variants_per_source + variant_index
                for source_index in chunk
                for variant_index in range(self.variants_per_source)
            ]
            for _ in range(self.repeat_count):
                for index in expanded_indices:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        full_source_count = (
            self.base_size // self.sources_per_batch
        ) * self.sources_per_batch
        return (
            full_source_count
            * self.variants_per_source
            * self.mini_repeat_count
            * self.repeat_count
        )
