import argparse
import json
import random
from pathlib import Path


# IDs that appear more than once in the dataset — suffixed with _1 / _2 to deduplicate.
_DUPLICATE_IDS = {
    "mathvista_117",
    "mathvista_241",
    "mathvista_313",
    "mathvista_498",
    "mathvista_629",
    "mathvista_751",
    "mathvista_963",
    "mathvista_974",
    "visitbench_286",
    "visitbench_312",
    "visitbench_544",
    "visitbench_61",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download syhuggingface/multimodal_rewardbench and convert to pairwise JSONL."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mmrewardbench",
        help="Directory to save the JSONL file and images/. (default: data/mmrewardbench)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="HuggingFace dataset split (default: test).",
    )
    parser.add_argument(
        "--paper_ids",
        type=str,
        default=None,
        help="Path to a JSON list of IDs (e.g. data/scripts/mmrewardbench_paper_ids.json). "
             "When set, filters to exactly those IDs in that order, ignoring --sample_size/--seed.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of examples to keep after filtering (default: 1000). Ignored if --paper_ids is set.",
    )
    parser.add_argument(
        "--no_subsample",
        action="store_true",
        help="Keep all examples (skip subsampling). Ignored if --paper_ids is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for subsampling (default: 123). Ignored if --paper_ids is set.",
    )
    return parser.parse_args()


def save_image(img, image_dir: Path, sample_id: str, idx: int) -> str:
    """Save a PIL image and return the absolute path string."""
    import numpy as np
    from PIL import Image

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        img = Image.open(img)

    image_path = image_dir / f"{sample_id}.png"
    if image_path.exists():
        image_path = image_dir / f"{sample_id}_{idx}.png"
    img.save(str(image_path))
    return str(image_path)


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            "Required packages missing. Install with:\n"
            "  pip install datasets tqdm Pillow"
        )

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading syhuggingface/multimodal_rewardbench (split={args.split}) ...")
    dataset = load_dataset("syhuggingface/multimodal_rewardbench", split=args.split) # you may download from the original repository: https://github.com/facebookresearch/multimodal_rewardbench
    print(f"  {len(dataset)} total examples")

    rows_raw = list(enumerate(tqdm(dataset, desc="Loading")))

    # Filter / subsample
    if args.paper_ids is not None:
        paper_ids = json.loads(Path(args.paper_ids).read_text(encoding="utf-8"))
        id_to_row = {sample["ID"]: (idx, sample) for idx, sample in rows_raw}
        missing = [i for i in paper_ids if i not in id_to_row]
        if missing:
            print(f"  WARNING: {len(missing)} paper IDs not found in dataset: {missing[:5]}")
        rows_raw = [id_to_row[i] for i in paper_ids if i in id_to_row]
        print(f"  Filtered to {len(rows_raw)} examples matching paper IDs")
    elif not args.no_subsample and args.sample_size > 0:
        if len(rows_raw) > args.sample_size:
            random.seed(args.seed)
            rows_raw = random.sample(rows_raw, args.sample_size)
            print(f"  Subsampled to {len(rows_raw)} examples (seed={args.seed})")

    # Convert and save images
    rows = []
    dup_counter: dict = {}
    for idx, sample in tqdm(rows_raw, desc="Saving images"):
        sample_id = sample["ID"]

        # Handle duplicate IDs by appending _1 / _2
        if sample_id in _DUPLICATE_IDS:
            dup_counter[sample_id] = dup_counter.get(sample_id, 0) + 1
            sample_id = f"{sample_id}_{dup_counter[sample_id]}"

        image_path = None
        if sample.get("Image") is not None:
            image_path = save_image(sample["Image"], image_dir, sample_id, idx)

        row = {
            "ID": sample_id,
            "Text": sample["Text"],
            "Output1": sample["Output1"],
            "Output2": sample["Output2"],
            "Better": sample["Better"],
            "Image": image_path,
        }
        for key in ("Category", "subset"):
            if key in sample and sample[key] is not None:
                row[key] = sample[key]
        rows.append(row)

    out_path = output_dir / f"mmrewardbench_{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved {len(rows)} rows → {out_path}")
    print(f"Images saved under: {image_dir}")
    if dup_counter:
        print(f"Deduplicated IDs: {list(dup_counter.keys())}")


if __name__ == "__main__":
    main()
