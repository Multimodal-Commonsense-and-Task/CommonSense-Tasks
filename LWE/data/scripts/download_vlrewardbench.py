import argparse
import json
from pathlib import Path

# IDs that appear more than once in the dataset — suffixed with _1 / _2 to deduplicate.
_DUPLICATE_IDS = {
    "mathverse_1649",
    "mathverse_1714",
    "mathverse_1719",
    "mathverse_1908",
    "mathverse_2013",
    "mathverse_2444",
    "mmmu_pro_test_Economics_16",
    "mmmu_pro_validation_Chemistry_17",
    "mmmu_pro_validation_Materials_7",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MMInstruction/VL-RewardBench and convert to pairwise JSONL."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/vlrewardbench",
        help="Directory to save the JSONL file and images/. (default: data/vlrewardbench)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="HuggingFace dataset split (default: test).",
    )
    return parser.parse_args()


def convert_example(example: dict, image_dir: Path, dup_counter: dict) -> dict:
    """
    Convert a raw HuggingFace row to the pairwise JSONL format.

    VL-RewardBench fields used:
      id              — unique sample identifier
      query           — the question / prompt
      response        — list[str] of length 2: [response_A, response_B]
      human_ranking   — list[int], e.g. [0, 1] means response_A is preferred
      image           — PIL Image
    """
    sample_id = example["id"]

    # Handle duplicate IDs by appending _1 / _2
    if sample_id in _DUPLICATE_IDS:
        dup_counter[sample_id] = dup_counter.get(sample_id, 0) + 1
        sample_id = sample_id + f"_{dup_counter[sample_id]}"

    # Save image
    image_path = None
    if example.get("image") is not None:
        img = example["image"].convert("RGB")
        save_path = image_dir / f"{sample_id}.jpg"
        if not save_path.exists():
            img.save(str(save_path))
        image_path = str(save_path)

    # human_ranking [0, 1] → response[0] is preferred → Output1 is better
    # human_ranking [1, 0] → response[1] is preferred → Output2 is better
    human_ranking = example.get("human_ranking", [0, 1])
    better = "Output1" if human_ranking[0] == 0 else "Output2"

    return {
        "ID": sample_id,
        "Text": example.get("query", ""),
        "Output1": example["response"][0],
        "Output2": example["response"][1],
        "Better": better,
        "Image": image_path,
    }


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

    print(f"Downloading MMInstruction/VL-RewardBench (split={args.split}) ...")
    dataset = load_dataset("MMInstruction/VL-RewardBench", split=args.split)
    print(f"  {len(dataset)} examples loaded")

    dup_counter: dict = {}
    rows = []
    for example in tqdm(dataset, desc="Converting"):
        rows.append(convert_example(example, image_dir, dup_counter))

    out_path = output_dir / f"vlrewardbench_{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved {len(rows)} rows → {out_path}")
    print(f"Images saved under: {image_dir}")
    if dup_counter:
        print(f"Deduplicated IDs: {list(dup_counter.keys())}")


if __name__ == "__main__":
    main()
