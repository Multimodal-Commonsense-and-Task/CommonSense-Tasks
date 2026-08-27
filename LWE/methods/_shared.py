"""Shared helpers for SSP / LWE (per-sample rubric generation + judging)."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict

from prompts import lwe_prompts as LP


def example_block(sample: Dict[str, Any]) -> str:
    return LP.STATIC_EXAMPLE_PLACEHOLDER.format(
        question=sample["Text"],
        answer_a=sample["Output1"],
        answer_b=sample["Output2"],
    )


def prompt_for_eval_generation(meta_text: str, sample: Dict[str, Any]) -> str:
    return (
        meta_text.strip()
        + "\n\n"
        + LP.STATIC_REQUIREMENTS_FOR_EVAL_PROMPT_GENERATION.strip()
        + "\n\n"
        + example_block(sample)
    )


def assemble_eval_prompt(
    generated_rubric: str,
    sample: Dict[str, Any],
    eval_requirements: str | None = None,
) -> str:
    reqs = (eval_requirements or LP.STATIC_REQUIREMENTS_FOR_EVAL_PROMPT).strip()
    return (
        generated_rubric.strip()
        + "\n\n"
        + example_block(sample)
        + "\n\n"
        + reqs
    )


def judgment_block(judgment: str) -> str:
    return LP.STATIC_JUDGMENT_PLACEHOLDER.format(judgment=judgment)


def meta_eval_prompt(meta_text: str, eval_prompt: str, judgment: str) -> str:
    """Matches `static_prompt_for_meta_eval` in the paper code (evolving_meta)."""
    return LP.STATIC_PROMPT_FOR_META_FEEDBACK.format(
        meta_prompt=meta_text,
        evaluation_prompt=eval_prompt,
        static_judgment_place_holder=judgment_block(judgment),
    )


def parse_meta_feedback_maybe(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def meta_feedback_to_str(meta_feedback: Any) -> str:
    if isinstance(meta_feedback, str):
        return meta_feedback
    return json.dumps(meta_feedback, ensure_ascii=False)


def format_batch_for_meta_update(records: list[dict[str, Any]]) -> str:
    """Each record uses keys `input`, `judgment`, `meta_feedback`."""
    chunks: list[str] = []
    for idx, r in enumerate(records):
        mf = meta_feedback_to_str(r["meta_feedback"])
        part = (
            f"[[{idx}-th Example]]\n{r['input']}\n\n"
            f"{LP.STATIC_JUDGMENT_PLACEHOLDER.format(judgment=r['judgment'])}\n\n"
            f"{LP.STATIC_META_FEEDBACK_PLACEHOLDER.format(meta_feedback=mf)}"
        )
        chunks.append(part.strip())
    return "\n\n".join(chunks)
