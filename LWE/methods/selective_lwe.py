"""
Selective-LWE: apply LWE only on samples where the vanilla judge is inconsistent.

Algorithm (per batch from the dataloader):
  1. Run vanilla judge with swap=True on every sample in the batch.
  2. Split results into:
       consistent   — vanilla original and swapped judgments agree (consistency == 1)
       inconsistent — vanilla judgments disagree    (consistency == 0)
  3. For inconsistent samples only: run the full LWE per-sample pipeline
     (rubric generation → judge with rubric → meta-eval → meta-feedback).
  4. Accumulate inconsistent samples in a buffer. Once the buffer reaches
     lwe_batch_size (default 4), update the meta-prompt and flush the buffer.
     Any remaining samples at the end of the dataset trigger a final update.

Consistent samples keep their vanilla judgment; inconsistent samples get the
LWE-refined judgment. Both are written to the same per-sample JSONL output.
"""

import json
import time
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, Dict, List

from methods import _shared as sh
from prompts import lwe_prompts as LP
from prompts.vanilla import VANILLA_JUDGE_PROMPT
from utils.utils import (
    compute_metrics,
    flip_label,
    format_prompt,
    log_cumulative_metrics,
    return_extract_judgment_fn,
    swap_sample,
    write_jsonl,
)

MIN_META_PROMPT_CHARS = 726 # when the generation fails and the meta prompt is too short, we revert to the previous meta prompt



def _vanilla_one(sample: Dict[str, Any], prompt_tmpl: str, extract_fn: Callable[[str], str], model, temperature: float) -> Dict[str, Any]:
    """Run vanilla judge (original + swap) for consistency checking."""
    prompt_orig = format_prompt(prompt_tmpl, sample)
    response = model.generate(prompt_orig, sample.get("Image"), temperature=temperature)

    sw = swap_sample(sample)
    prompt_swap = format_prompt(prompt_tmpl, sw)
    swap_response = model.generate(prompt_swap, sw.get("Image"), temperature=temperature)

    pred = extract_fn(response)
    swap_pred = extract_fn(swap_response)
    consistent = int(
        pred in ("A", "B") and swap_pred in ("A", "B") and pred == flip_label(swap_pred)
    )

    return {
        **sample,
        "prompt": prompt_orig,
        "response": response,
        "swap_response": swap_response,
        "pred": pred,
        "swap_pred": swap_pred,
        "consistency": consistent,
    }


def _lwe_one(
    sample: Dict[str, Any],
    meta_prompt: str,
    model,
    temperature: float,
    eval_requirements: str | None = None,
) -> Dict[str, Any]:
    """Run LWE pipeline (rubric → judge → meta-eval) for one inconsistent sample."""
    gen_in = sh.prompt_for_eval_generation(meta_prompt, sample)
    rubric = model.generate(gen_in, sample.get("Image"), temperature=temperature)
    eval_prompt = sh.assemble_eval_prompt(rubric, sample, eval_requirements=eval_requirements)
    judgment = model.generate(eval_prompt, sample.get("Image"), temperature=temperature)

    meta_eval_input = sh.meta_eval_prompt(meta_prompt, eval_prompt, judgment)
    meta_feedback_raw = model.generate(meta_eval_input, sample.get("Image"), temperature=temperature)
    meta_feedback = sh.parse_meta_feedback_maybe(meta_feedback_raw)

    return {
        **sample,
        "meta_prompt": meta_prompt,
        "rubric_generation_input": gen_in,
        "generated_rubric": rubric,
        "eval_prompt": eval_prompt,
        "response": judgment,
        "meta_eval_input": meta_eval_input,
        "meta_feedback_raw": meta_feedback_raw,
        "meta_feedback": meta_feedback,
    }


def _update_meta_prompt(
    meta_prompt: str,
    buffer: List[Dict[str, Any]],
    model,
    temperature: float,
    restrict_length: bool,
    max_meta_len: int,
) -> str:
    meta_prompt_update_template = (
        LP.STATIC_PROMPT_FOR_META_PROMPT_UPDATE_RESTRICT_LENGTH
        if restrict_length
        else LP.STATIC_PROMPT_FOR_META_PROMPT_UPDATE
    )
    update_lines = []
    for row in buffer:
        mf = row["meta_feedback"]
        if not isinstance(mf, str):
            mf = json.dumps(mf, ensure_ascii=False)
        update_lines.append(
            {
                "input": row["eval_prompt"],
                "judgment": row["response"],
                "meta_feedback": mf,
                "Image": row.get("Image"),
            }
        )
    batch_str = sh.format_batch_for_meta_update(update_lines)
    meta_prompt_update_prompt = meta_prompt_update_template.format(meta_prompt=meta_prompt, batch=batch_str)
    images = [r.get("Image") for r in update_lines]
    response = model.generate(meta_prompt_update_prompt, image=images, temperature=temperature)

    if restrict_length and len(response) > max_meta_len:
        summarize_prompt = LP.SUMMARIZE_META_PROMPT.format(meta_prompt=response)
        response = model.generate(summarize_prompt, image=None, temperature=temperature)
    return response


def run_dataset(model, dataloader, cfg: Dict[str, Any]) -> None:
    _start = time.time()
    run_dir = Path(cfg["run_dir"])
    ds_name = cfg["dataset"]["name"]
    per_sample_path = run_dir / f"{ds_name}.jsonl"
    meta_dir = run_dir / "meta_prompt_snapshots"
    meta_dir.mkdir(parents=True, exist_ok=True)

    lwe_cfg = cfg.get("lwe", {})
    meta_prompt = (
        cfg.get("prompts", {}).get("initial_meta_prompt") or LP.INITIAL_META_PROMPT
    )
    version = 0
    (run_dir / "meta_prompt_v0_initial.txt").write_text(meta_prompt, encoding="utf-8")
    (meta_dir / "v0_meta_prompt.txt").write_text(meta_prompt, encoding="utf-8")

    vanilla_prompt_tmpl = (
        cfg.get("prompts", {}).get("vanilla_judge_prompt") or VANILLA_JUDGE_PROMPT
    )
    eval_requirements = cfg.get("prompts", {}).get("eval_requirements")
    extract_fn = return_extract_judgment_fn(ds_name)
    temperature = float(cfg.get("temperature", 0.0))
    restrict_length = bool(lwe_cfg.get("restrict_length", False))
    max_meta_len = int(lwe_cfg.get("max_meta_prompt_length", 10000))
    lwe_batch_size = int(lwe_cfg.get("lwe_batch_size", 4))

    cumulative: Dict[str, List[float]] = {"acc": [], "swap_acc": [], "consistency": [], "pair_acc": []}
    all_results: List[Dict[str, Any]] = []
    lwe_buffer: List[Dict[str, Any]] = []  # accumulates inconsistent samples across dataloader batches
    batch_idx = 0
    meta_update_idx = 0

    def _maybe_update_meta_prompt(force: bool = False) -> None:
        nonlocal meta_prompt, version, meta_update_idx
        while len(lwe_buffer) >= lwe_batch_size or (force and lwe_buffer):
            chunk = lwe_buffer[:lwe_batch_size]
            del lwe_buffer[:lwe_batch_size]
            prev_meta_prompt = meta_prompt
            meta_prompt = _update_meta_prompt(
                meta_prompt, chunk, model, temperature, restrict_length, max_meta_len
            )
            if len(meta_prompt) < MIN_META_PROMPT_CHARS:
                print(
                    f"[selective_lwe] Meta update too short ({len(meta_prompt)} chars); reverting."
                )
                meta_prompt = prev_meta_prompt
            else:
                version += 1
                (meta_dir / f"v{meta_update_idx + 1}_meta_prompt.txt").write_text(
                    meta_prompt, encoding="utf-8"
                )
            meta_update_idx += 1

    for batch in dataloader:
        t0 = time.time()

        # --- Step 1: vanilla pass with swap (consistency check) ---
        vanilla_worker = partial(
            _vanilla_one,
            prompt_tmpl=vanilla_prompt_tmpl,
            extract_fn=extract_fn,
            model=model,
            temperature=temperature,
        )
        with ThreadPool(len(batch)) as pool:
            vanilla_results = list(pool.imap(vanilla_worker, batch))

        consistent = [r for r in vanilla_results if r["consistency"] == 1]
        inconsistent = [r for r in vanilla_results if r["consistency"] == 0]

        print(
            f"[selective_lwe] batch {batch_idx}: "
            f"{len(consistent)} consistent, {len(inconsistent)} inconsistent"
        )

        # --- Step 2: LWE pass for inconsistent samples ---
        lwe_results: List[Dict[str, Any]] = []
        if inconsistent:
            lwe_worker = partial(
                _lwe_one,
                meta_prompt=meta_prompt,
                model=model,
                temperature=temperature,
                eval_requirements=eval_requirements,
            )
            with ThreadPool(len(inconsistent)) as pool:
                lwe_results = list(pool.imap(lwe_worker, inconsistent))
            lwe_buffer.extend(lwe_results)

        # --- Step 3: update meta-prompt once buffer reaches lwe_batch_size ---
        _maybe_update_meta_prompt(force=False)

        # --- Step 4: merge and log ---
        merged = consistent + lwe_results
        for row in merged:
            row["meta_prompt_after_batch"] = meta_prompt
            row["meta_prompt_version"] = version

        all_results.extend(merged)
        write_jsonl(per_sample_path, all_results)
        cumulative = compute_metrics(merged, cumulative, extract_fn)
        log_cumulative_metrics(run_dir, cumulative)
        print(
            f"[selective_lwe] batch {batch_idx} time: {time.time() - t0:.2f}s, "
            f"n={len(merged)}, meta_v={version}"
        )
        batch_idx += 1

    # flush any remaining inconsistent samples that didn't fill a full lwe_batch_size
    _maybe_update_meta_prompt(force=True)

    (run_dir / "meta_prompt_final.txt").write_text(meta_prompt, encoding="utf-8")
    print(f"Done. samples={len(all_results)} time={time.time() - _start:.1f}s")
