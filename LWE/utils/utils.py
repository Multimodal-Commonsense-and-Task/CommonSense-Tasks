import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Union[str, Path], data: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def flip_label(label: str) -> str:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    if label == "Output1":
        return "Output2"
    if label == "Output2":
        return "Output1"
    return label


def swap_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(sample)
    s["Output1"], s["Output2"] = sample["Output2"], sample["Output1"]
    if "Better" in s:
        s["Better"] = flip_label(s["Better"])
    return s


def format_prompt(prompt_tmpl: str, sample: Dict[str, Any]) -> str:
    return prompt_tmpl.format(
        question=sample["Text"],
        answer_a=sample["Output1"],
        answer_b=sample["Output2"],
    )


def extract_judgment_mmrewardbench(judgment: str) -> str:
    if "[[A]]" in judgment and "[[B]]" in judgment:
        return "Not judged in the proper format.  [[A,B]]"
    if "[[A]]" in judgment:
        return "A"
    if "[[B]]" in judgment:
        return "B"
    if "[A]" in judgment:
        return "A"
    if "[B]" in judgment:
        return "B"
    return "Not judged in the proper format."


def return_extract_judgment_fn(benchmark: str) -> Callable[[str], str]:
    if benchmark in ("vlrewardbench", "mmrewardbench"):
        return extract_judgment_mmrewardbench
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def compute_metrics(
    batch: Sequence[Dict[str, Any]],
    cumulative_metrics: Dict[str, List[int]],
    extract_judgment: Callable[[str], str],
) -> Dict[str, List[int]]:
    for example in batch:
        example["pred"] = extract_judgment(example["response"])
        label = "A" if example["Better"] == "Output1" else "B"
        example["acc"] = int(label == example["pred"])
        cumulative_metrics["acc"].append(example["acc"])

        if "swap_response" in example:
            example["swap_pred"] = extract_judgment(example["swap_response"])
            example["swap_acc"] = int(label == flip_label(example["swap_pred"]))
            cumulative_metrics["swap_acc"].append(example["swap_acc"])

            if example["pred"] in ["A", "B"] and example["swap_pred"] in ["A", "B"]:
                consistency = int(example["pred"] == flip_label(example["swap_pred"]))
            else:
                consistency = 0
            example["consistency"] = consistency
            cumulative_metrics["consistency"].append(consistency)

            pair_acc = int(example["acc"] == 1 and example["swap_acc"] == 1)
            example["pair_acc"] = pair_acc
            cumulative_metrics["pair_acc"].append(pair_acc)
    return cumulative_metrics


def log_cumulative_metrics(run_dir: Path, cumulative_metrics: Dict[str, List[int]]) -> None:
    results: Dict[str, Any] = {
        "acc": {
            "sum": sum(cumulative_metrics["acc"]),
            "count": len(cumulative_metrics["acc"]),
            "mean": round(
                sum(cumulative_metrics["acc"]) / len(cumulative_metrics["acc"]), 3
            ),
        }
    }
    if cumulative_metrics.get("swap_acc"):
        results["swap_acc"] = {
            "sum": sum(cumulative_metrics["swap_acc"]),
            "count": len(cumulative_metrics["swap_acc"]),
            "mean": round(
                sum(cumulative_metrics["swap_acc"])
                / len(cumulative_metrics["swap_acc"]),
                3,
            ),
        }
    if cumulative_metrics.get("consistency"):
        results["consistency"] = {
            "sum": sum(cumulative_metrics["consistency"]),
            "count": len(cumulative_metrics["consistency"]),
            "mean": round(
                sum(cumulative_metrics["consistency"])
                / len(cumulative_metrics["consistency"]),
                3,
            ),
        }
    if cumulative_metrics.get("pair_acc"):
        results["pair_acc"] = {
            "sum": sum(cumulative_metrics["pair_acc"]),
            "count": len(cumulative_metrics["pair_acc"]),
            "mean": round(
                sum(cumulative_metrics["pair_acc"])
                / len(cumulative_metrics["pair_acc"]),
                4,
            ),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "cumulative_metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")
