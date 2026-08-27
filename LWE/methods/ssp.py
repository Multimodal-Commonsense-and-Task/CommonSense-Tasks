"""
SSP (sample-specific prompt) — static meta-prompt.

The **meta-prompt** (instruction to produce a rubric) is fixed across the dataset.
For each sample we:
  1) generate a sample-specific evaluation rubric from the meta-prompt,
  2) attach the pairwise instance + output-format constraints,
  3) run the judge once to get [[A]]/[[B]].
"""

import time
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List

from prompts import lwe_prompts as LP
from methods import _shared as sh
from utils.utils import (
    compute_metrics,
    log_cumulative_metrics,
    return_extract_judgment_fn,
    swap_sample,
    write_jsonl,
)


def _one_sample(
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

    row: Dict[str, Any] = {
        **sample,
        "meta_prompt": meta_prompt,
        "rubric_generation_input": gen_in,
        "generated_rubric": rubric,
        "eval_prompt": eval_prompt,
        "response": judgment,
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

    meta_prompt = (
        cfg.get("prompts", {}).get("initial_meta_prompt")
        or LP.INITIAL_META_PROMPT
    )
    extract_fn = return_extract_judgment_fn(ds_name)
    run_swap = cfg.get("run_swap", False)
    temperature = float(cfg.get("temperature", 0.0))

    cumulative = {"acc": [], "swap_acc": [], "consistency": [], "pair_acc": []}
    all_results: List[Dict[str, Any]] = []

    worker = partial(
        _one_sample,
        meta_prompt=meta_prompt,
        model=model,
        temperature=temperature,
        run_swap=run_swap,
    )

    for batch in dataloader:
        t0 = time.time()
        with ThreadPool(len(batch)) as pool:
            merged = list(pool.imap(worker, batch))
        all_results.extend(merged)
        write_jsonl(per_sample_path, all_results)
        cumulative = compute_metrics(merged, cumulative, extract_fn)
        log_cumulative_metrics(run_dir, cumulative)
        print(f"[ssp] batch time: {time.time() - t0:.2f}s, n={len(merged)}")

    (run_dir / "meta_prompt_initial.txt").write_text(meta_prompt, encoding="utf-8")
    print(f"Done. samples={len(all_results)} time={time.time() - _start:.1f}s")
