#!/usr/bin/env python3
"""Entrypoint: `python judge.py --config configs/vanilla.yaml`"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
import random
import time
from typing import Any, Dict

import yaml
from torch.utils.data import DataLoader

# Repo root (this file's directory) must be on sys.path for `data`, `methods`, …
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.dataset import PairwiseDataset, create_dataloader
from models.base import BaseModel


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict. Got {type(cfg)} from {path}")
    return cfg


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_run_dir(base: str | Path, method: str, model_name: str) -> Path:
    base = Path(base)
    safe_model = model_name.replace("/", "_").replace(":", "_")
    run_name = f"{method}_{safe_model}_{_now_ts()}"
    return ensure_dir(base / run_name)


def run_judge(
    method: str, model_obj: BaseModel, dataloader: DataLoader, cfg: Dict[str, Any]
) -> None:
    if method == "vanilla":
        from methods.vanilla import run_dataset
    elif method == "ssp":
        from methods.ssp import run_dataset
    elif method == "lwe":
        from methods.lwe import run_dataset
    elif method == "selective_lwe":
        from methods.selective_lwe import run_dataset
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from "
            "[vanilla, ssp, lwe, selective_lwe]."
        )
    run_dataset(model_obj, dataloader, cfg)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judge.py",
        description="Becoming Experienced Judges — vanilla / SSP / LWE / Selective LWE.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--method", type=str, default=None, help="vanilla | ssp | lwe | selective_lwe"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model_name (nested under config['model']).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    return parser


def load_model(model_config: Dict[str, Any]) -> BaseModel:
    model_type = model_config["model_type"]
    if model_type == "gpt":
        from models.gpt import GPTModel
        return GPTModel(
            model_name=model_config["model_name"],
            max_try=model_config.get("max_try", 10),
        )
    if model_type == "gemini":
        from models.gemini import GeminiModel
        return GeminiModel(
            model_name=model_config["model_name"],
            max_try=model_config.get("max_try", 10),
        )
    if model_type == "claude":
        from models.claude import ClaudeModel
        return ClaudeModel(
            model_name=model_config["model_name"],
            max_try=model_config.get("max_try", 10),
            max_tokens=model_config.get("max_tokens", 62000),
        )
    raise ValueError(
        f"Unknown model type '{model_type}'. Choose from [gpt, gemini, claude]."
    )


def append_img_prefix(data: list, image_prefix: str) -> list:
    for x in data:
        if "Image" in x and x["Image"] is not None:
            x["Image"] = image_prefix + x["Image"]
    return data


def main():
    args = build_argparser().parse_args()
    config = load_yaml(args.config)

    if args.method is not None:
        config["method"] = args.method
    if args.seed is not None:
        config["seed"] = args.seed
    if args.model is not None:
        m = config.get("model")
        if isinstance(m, dict):
            m["model_name"] = args.model
        else:
            raise ValueError(
                "Config must define `model` as a dict with model_type / model_name "
                "when using --model override."
            )

    assert config["method"] in (
        "vanilla",
        "ssp",
        "lwe",
        "selective_lwe",
    ), "Invalid method."
    assert config.get("model") is not None, "Config must set `model`."
    assert config.get("dataset") is not None, "Config must set `dataset`."

    method = config["method"]
    model_config = config["model"]
    seed = int(config.get("seed", 0))
    dataset_config = config["dataset"]

    set_global_seed(seed)

    run_dir = make_run_dir(
        config.get("out_dir", "runs"), method, model_config["model_name"]
    )
    config["run_dir"] = str(run_dir)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(
        f"[Run] method={method} model={model_config['model_name']} "
        f"temp={config.get('temperature', 0.0)} seed={seed}"
    )
    print(f"[Run] run_dir={run_dir}")

    proc = partial(
        append_img_prefix, image_prefix=dataset_config.get("image_prefix") or ""
    )
    dataset = PairwiseDataset(
        data_path=dataset_config["data_path"],
        swap=dataset_config.get("swap", False),
        processing_func=proc if dataset_config.get("image_prefix") else None,
        data_start_idx=dataset_config.get("data_start_idx"),
        data_end_idx=dataset_config.get("data_end_idx"),
        order_ids_path=dataset_config.get("order_ids_path"),
        strict_order=dataset_config.get("strict_order", True),
    )
    dataloader = create_dataloader(
        dataset,
        dataset_config.get("batch_size", 4),
        dataset_config.get("shuffle", False),
    )

    model_obj = load_model(model_config)
    run_judge(method, model_obj, dataloader, config)


if __name__ == "__main__":
    main()
