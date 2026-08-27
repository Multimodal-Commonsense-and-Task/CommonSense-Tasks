"""
LWE — meta-prompt evolution (paper: learning from experience at test time).

Per sample:
  1) Generate sample-specific rubric from the **current** meta-prompt (same pipeline as SSP).
  2) Judge with rubric + instance + output format.
  3) Meta-eval: critique judgment and emit structured feedback for meta-prompt improvement.

After each dataloader batch:
  4) Batch-update the meta-prompt from (eval_prompt, judgment, meta_feedback, images).
"""

import json
import time
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List

from methods import _shared as sh
from prompts import lwe_prompts as LP
from utils.utils import (
    compute_metrics,
    log_cumulative_metrics,
    return_extract_judgment_fn,
    swap_sample,
    write_jsonl,
)

MIN_META_PROMPT_CHARS = 726  # when the generation fails and the meta prompt is too short, we revert to the previous meta prompt


def _update_meta_prompt(
    meta_text: str,
    batch_records: List[Dict[str, Any]],
    model,
    temperature: float,
    restrict_length: bool,
    max_meta_len: int,
) -> str:
    tmpl = (
        LP.STATIC_PROMPT_FOR_META_PROMPT_UPDATE_RESTRICT_LENGTH
        if restrict_length
        else LP.STATIC_PROMPT_FOR_META_PROMPT_UPDATE
    )
    batch_str = sh.format_batch_for_meta_update(batch_records)
    prompt = tmpl.format(meta_prompt=meta_text, batch=batch_str)
    images = [r.get("Image") for r in batch_records]
    response = model.generate(prompt, image=images, temperature=temperature)

    if restrict_length and len(response) > max_meta_len:
        sp = LP.SUMMARIZE_META_PROMPT.format(meta_prompt=response)
        response = model.generate(sp, image=None, temperature=temperature)
    return response


def _one_sample_lwe(
    sample: Dict[str, Any],
    meta_prompt: str,
    model,
    temperature: float,
    run_swap: bool,
) -> Dict[str, Any]:
    gen_in = sh.prompt_for_eval_generation(meta_prompt, sample)
    rubric = model.generate(gen_in, sample.get("Image"), temperature=temperature)
    eval_prompt = sh.assemble_eval_prompt(rubric, sample)
    judgment = model.generate(eval_prompt, sample.get("Image"), temperature=temperature)

    meta_eval_input = sh.meta_eval_prompt(meta_prompt, eval_prompt, judgment)
    meta_feedback_raw = model.generate(meta_eval_input, sample.get("Image"), temperature=temperature)
    meta_feedback = sh.parse_meta_feedback_maybe(meta_feedback_raw)

    row: Dict[str, Any] = {
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

    if run_swap:
        sw = swap_sample(sample)
        gen_in_s = sh.prompt_for_eval_generation(meta_prompt, sw)
        rubric_s = model.generate(gen_in_s, sw.get("Image"), temperature=temperature)
        eval_prompt_s = sh.assemble_eval_prompt(rubric_s, sw)
        row["swap_response"] = model.generate(
            eval_prompt_s, sw.get("Image"), temperature=temperature
        )

    return row


def run_dataset(model, dataloader, cfg: Dict[str, Any]) -> None:
    _start = time.time()
    run_dir = Path(cfg["run_dir"])
    ds_name = cfg["dataset"]["name"]
    per_sample_path = run_dir / f"{ds_name}.jsonl"
    meta_dir = run_dir / "meta_prompt_snapshots"
    meta_dir.mkdir(parents=True, exist_ok=True)

    lwe_cfg = cfg.get("lwe", {})
    meta_prompt = (
        cfg.get("prompts", {}).get("initial_meta_prompt")
        or LP.INITIAL_META_PROMPT
    )
    version = 0
    (run_dir / "meta_prompt_v0_initial.txt").write_text(meta_prompt, encoding="utf-8")
    (meta_dir / "v0_meta_prompt.txt").write_text(meta_prompt, encoding="utf-8")

    extract_fn = return_extract_judgment_fn(ds_name)
    run_swap = cfg.get("run_swap", False)
    temperature = float(cfg.get("temperature", 0.0))
    restrict_length = bool(lwe_cfg.get("restrict_length", False))
    max_meta_len = int(lwe_cfg.get("max_meta_prompt_length", 10000))

    cumulative = {"acc": [], "swap_acc": [], "consistency": [], "pair_acc": []}
    all_results: List[Dict[str, Any]] = []
    batch_idx = 0

    for batch in dataloader:
        t0 = time.time()
        worker = partial(
            _one_sample_lwe,
            meta_prompt=meta_prompt,
            model=model,
            temperature=temperature,
            run_swap=run_swap,
        )
        with ThreadPool(len(batch)) as pool:
            merged = list(pool.imap(worker, batch))

        update_lines = []
        for row in merged:
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

        # update the meta prompt per batch
        prev_meta_prompt = meta_prompt
        meta_prompt = _update_meta_prompt(
            meta_prompt,
            update_lines,
            model,
            temperature,
            restrict_length,
            max_meta_len,
        )
        if len(meta_prompt) < MIN_META_PROMPT_CHARS:
            print(
                f"[lwe] Meta update too short ({len(meta_prompt)} chars); reverting to previous."
            )
            meta_prompt = prev_meta_prompt
        else:
            version += 1
            (meta_dir / f"v{batch_idx + 1}_meta_prompt.txt").write_text(
                meta_prompt, encoding="utf-8"
            )

        for row in merged:
            row["meta_prompt_after_batch"] = meta_prompt
            row["meta_prompt_version"] = version

        all_results.extend(merged)
        write_jsonl(per_sample_path, all_results)
        cumulative = compute_metrics(merged, cumulative, extract_fn)
        log_cumulative_metrics(run_dir, cumulative)
        print(
            f"[lwe] batch {batch_idx} time: {time.time() - t0:.2f}s, n={len(merged)}, meta_v={version}"
        )
        batch_idx += 1

    (run_dir / "meta_prompt_final.txt").write_text(meta_prompt, encoding="utf-8")
    print(f"Done. samples={len(all_results)} time={time.time() - _start:.1f}s")
