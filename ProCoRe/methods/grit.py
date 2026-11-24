"""GRIT prompt shaping and rewards adapted from the official release.

The reference implementation is in ``/home/ubuntu/mjlee/GRIT/grpo-gr``.
Its one-turn recipe uses ordinary clipped GRPO together with a grounded
``think -> bounding boxes -> rethink -> answer`` response and six summed
reward components. This module keeps those components dependency-free so
LMM-ICL does not require GRIT's private Azure endpoint.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any, Sequence

from torch.utils.data import Dataset


GRIT_PROMPT_SUFFIX = """
First, reason between <think> and </think>. Inside <think>, output every image
region needed to answer using integer bounding boxes in the form
{"bbox_2d": [x1, y1, x2, y2], "label": "object"}, with coordinates normalized
to the 0-1000 range. Then reconsider the evidence between <rethink> and
</rethink>. Finally, put only the requested answer between <answer> and
</answer>.
""".strip()

_BBOX_RE = re.compile(r"\b\d+,\s*\d+,\s*\d+,\s*\d+\b")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _append_instruction(prompt: Any) -> Any:
    """Append the GRIT response contract to a string or chat prompt."""
    prompt = copy.deepcopy(prompt)
    if isinstance(prompt, str):
        return f"{prompt.rstrip()}\n{GRIT_PROMPT_SUFFIX}"
    if not isinstance(prompt, list):
        raise TypeError(f"Unsupported GRIT prompt type: {type(prompt).__name__}")

    for message in reversed(prompt):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = f"{content.rstrip()}\n{GRIT_PROMPT_SUFFIX}"
            return prompt
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = (
                        f"{str(block.get('text', '')).rstrip()}\n"
                        f"{GRIT_PROMPT_SUFFIX}"
                    )
                    return prompt
            content.append({"type": "text", "text": GRIT_PROMPT_SUFFIX})
            return prompt

    raise ValueError("GRIT samples must contain a user prompt")


class GRITPromptDataset(Dataset):
    """View of an existing GRPO dataset with the GRIT prompt appended."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Copy the mapping but not the potentially large PIL image objects.
        sample = dict(self.dataset[index])
        sample["prompt"] = _append_instruction(sample["prompt"])
        return sample


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            content = item.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
    return str(completion or "")


def _answer_text(text: str) -> str:
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    if "<answer>" in text.lower():
        return re.split(r"<answer>", text, flags=re.IGNORECASE)[-1].strip()
    return text.strip()


def _tokens(text: Any) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l(prediction: Any, reference: Any) -> float:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(prediction_tokens, reference_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _bleu1(prediction: Any, reference: Any) -> float:
    """Dependency-free BLEU-1 matching NLTK's clipped precision and BP."""
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    prediction_counts = Counter(prediction_tokens)
    reference_counts = Counter(reference_tokens)
    clipped = sum(
        min(count, reference_counts[token])
        for token, count in prediction_counts.items()
    )
    precision = clipped / len(prediction_tokens)
    if precision == 0.0:
        return 0.0
    brevity_penalty = (
        1.0
        if len(prediction_tokens) > len(reference_tokens)
        else math.exp(1.0 - len(reference_tokens) / len(prediction_tokens))
    )
    return brevity_penalty * precision


def _caption_pair(answer: str) -> tuple[str, str]:
    parts = answer.split("||", maxsplit=1)
    if len(parts) == 1:
        return parts[0].strip(), ""
    return parts[0].strip(), parts[1].strip()


def _paired_score(
    completion: Any,
    positive_gold: Any,
    negative_gold: Any,
    score_fn,
) -> float:
    answer = _answer_text(_completion_text(completion))
    if negative_gold is None:
        return score_fn(answer, positive_gold)
    positive_prediction, negative_prediction = _caption_pair(answer)
    return (
        score_fn(positive_prediction, positive_gold)
        + score_fn(negative_prediction, negative_gold)
    ) / 2.0


def grit_answer_reward(completions, golds, **kwargs):
    """Task outcome reward; paired ROUGE-L replaces GRIT's private GPT judge."""
    negative_golds = kwargs.get("gt_neg") or [None] * len(golds)
    return [
        _paired_score(completion, positive, negative, _rouge_l)
        for completion, positive, negative in zip(
            completions, golds, negative_golds
        )
    ]


def grit_bleu_reward(completions, golds, **kwargs):
    """Official GRIT BLEU-1 auxiliary reward, capped at a 0.1 scale."""
    negative_golds = kwargs.get("gt_neg") or [None] * len(golds)
    return [
        0.1 * _paired_score(completion, positive, negative, _bleu1)
        for completion, positive, negative in zip(
            completions, golds, negative_golds
        )
    ]


def grit_answer_format_reward(completions, **kwargs):
    """Official answer-tag reward with a maximum value of 0.5."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        occurrences = text.count("<answer>")
        reward = 0.0
        if occurrences:
            reward += 1.0
            if occurrences == 1:
                reward += 1.0
        rewards.append(reward / 2.0 * 0.5)
    return rewards


def grit_grounding_format_reward(completions, **kwargs):
    """Reward at least one integer bounding box before the rethink section."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        if "<rethink>" not in text:
            rewards.append(0.0)
            continue
        think_text = text.split("<rethink>", maxsplit=1)[0]
        rewards.append(0.5 if _BBOX_RE.search(think_text) else 0.0)
    return rewards


def grit_think_rethink_format_reward(completions, **kwargs):
    """Official ordered think/rethink tag reward with a maximum value of 0.5."""
    keys = ("<think>", "</think>", "<rethink>", "</rethink>")
    rewards = []
    for completion in completions:
        original = _completion_text(completion)
        remaining = original
        score = 0.0
        for key in keys:
            if key in remaining:
                score += 1.0
                remaining = remaining.split(key, maxsplit=1)[-1]
        if score == len(keys):
            try:
                thinking = original.split("<think>", maxsplit=1)[1].split(
                    "</think>", maxsplit=1
                )[0]
                if len(_tokens(thinking)) > 1:
                    score += 1.0
            except (IndexError, ValueError):
                pass
        rewards.append(score / (len(keys) + 1) * 0.5)
    return rewards


def _adjacent_repeat_fraction(items: Sequence[Any], ngram_size: int = 8) -> float:
    if len(items) < 2 * ngram_size:
        return 0.0
    pairs = 0
    repeats = 0
    for index in range(len(items) - 2 * ngram_size + 1):
        left = tuple(items[index : index + ngram_size])
        right = tuple(items[index + ngram_size : index + 2 * ngram_size])
        pairs += 1
        repeats += int(left == right)
    return repeats / pairs if pairs else 0.0


def grit_repetition_reward(completions, completion_ids=None, **kwargs):
    """Official word/token adjacent-repeat reward on a 0.5 scale."""
    completion_ids = (
        completion_ids
        if completion_ids is not None
        else [None] * len(completions)
    )
    rewards = []
    for completion, token_ids in zip(completions, completion_ids):
        text = _completion_text(completion)
        words = text.split()
        if not text or len(words) < 8:
            rewards.append(0.5)
            continue

        word_repeat = _adjacent_repeat_fraction(words)
        if token_ids is None:
            token_repeat = word_repeat
        else:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            token_repeat = _adjacent_repeat_fraction(list(token_ids))
        rewards.append((1.0 - token_repeat - word_repeat) * 0.5)
    return rewards


GRIT_REWARD_FUNCTIONS = (
    grit_answer_reward,
    grit_bleu_reward,
    grit_answer_format_reward,
    grit_repetition_reward,
    grit_grounding_format_reward,
    grit_think_rethink_format_reward,
)
