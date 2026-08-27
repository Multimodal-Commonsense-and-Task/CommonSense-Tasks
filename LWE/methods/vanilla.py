"""Vanilla LLM judge: one fixed prompt template, one generation per sample."""

import time
from pathlib import Path
from typing import Any, Dict, List

from prompts.vanilla import VANILLA_JUDGE_PROMPT
from utils.utils import (
    compute_metrics,
    format_prompt,
    log_cumulative_metrics,
    return_extract_judgment_fn,
    swap_sample,
    write_jsonl,
)


def run_dataset(model, dataloader, cfg: Dict[str, Any]) -> None:
    _start = time.time()
    run_dir = Path(cfg["run_dir"])
    ds_name = cfg["dataset"]["name"]
    per_sample_path = run_dir / f"{ds_name}.jsonl"

    prompt_tmpl = cfg.get("prompts", {}).get("vanilla_judge_prompt", VANILLA_JUDGE_PROMPT)
    extract_fn = return_extract_judgment_fn(ds_name)
    run_swap = cfg.get("run_swap", False)
    temperature = float(cfg.get("temperature", 0.0))

    cumulative = {"acc": [], "swap_acc": [], "consistency": [], "pair_acc": []}
    all_results: List[Dict[str, Any]] = []

    for batch in dataloader:
        t0 = time.time()
        input_batch = [
            {
                "Text": format_prompt(prompt_tmpl, sample),
                "Image": sample.get("Image"),
                "idx": idx,
            }
            for idx, sample in enumerate(batch)
        ]
        batch_responses = model.generate_batch(input_batch, temperature=temperature)

        merged: List[Dict[str, Any]] = []
        for r in batch_responses:
            sample = batch[r["idx"]]
            row = {
                **sample,
                "prompt": r["prompt"],
                "response": r["response"],
            }
            if run_swap:
                sw = swap_sample(sample)
                row["swap_response"] = model.generate(
                    format_prompt(prompt_tmpl, sw),
                    sw.get("Image"),
                    temperature=temperature,
                )
            merged.append(row)

        all_results.extend(merged)
        write_jsonl(per_sample_path, all_results)
        cumulative = compute_metrics(merged, cumulative, extract_fn)
        log_cumulative_metrics(run_dir, cumulative)
        print(f"[vanilla] batch time: {time.time() - t0:.2f}s, n={len(merged)}")

    print(f"Done. samples={len(all_results)} time={time.time() - _start:.1f}s")
