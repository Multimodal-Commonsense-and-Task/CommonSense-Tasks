set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_REPO_ROOT}}}"

if [[ "${REPO_ROOT}" == /var/spool/* ]]; then
    REPO_ROOT="${SCRIPT_REPO_ROOT}"
fi

cd "${REPO_ROOT}"

# GPU id(s) for the eval, e.g. `bash eval_ours2.sh 4` or `bash eval_ours2.sh 4,5`.
# Falls back to the GPU_ID env var, then to 0.
GPU_ID="${GPU_ID:-${1:-3}}"

# Models to evaluate, in order. Override with e.g. MODELS="qwen3-4b qwen3-8b".
# Available: onevision1_5-8b, qwen3-2b, qwen3-4b, qwen3-8b, internvl3_5-8b, idefics-8b, otter-9b
read -r -a MODELS <<< "${MODELS:-qwen3-2b qwen3-4b}"
BENCHMARK_FILTER="${BENCHMARK_FILTER:-mic}"
ACTIVATION_ITER="${MTV_ACTIVATION_ITER:-${MTV_FAST_FEATURES:+50}}"
ACTIVATION_ITER="${ACTIVATION_ITER:-100}"
ACTIVATION_SHOT=4
REINFORCE_EXAMPLES="${MTV_REINFORCE_EXAMPLES:-${MTV_FAST_FEATURES:+50}}"
REINFORCE_EXAMPLES="${REINFORCE_EXAMPLES:-100}"
export ACTIVATION_ITER ACTIVATION_SHOT REINFORCE_EXAMPLES

JSON_DIR="${REPO_ROOT}/MTV/existing_benchmark_json"
mkdir -p "${JSON_DIR}"

run_benchmark() {
    local dataset_name="$1"
    local activation_json="${JSON_DIR}/mtv_activation_${dataset_name}.json"
    local reinforce_json="${JSON_DIR}/mtv_reinforce_${dataset_name}.json"
    local storage_dir="${REPO_ROOT}/MTV/storage/${MODEL}/${dataset_name}"

    mkdir -p "${storage_dir}"

    python3 - "${dataset_name}" "${activation_json}" "${reinforce_json}" <<'PY' || return 1
import json
import os
import random
import sys
from pathlib import Path

DATASET_NAME = sys.argv[1]
ACTIVATION_JSON = Path(sys.argv[2])
REINFORCE_JSON = Path(sys.argv[3])

STANDARD_CONTENT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

RNG = random.Random(0)
ACTIVATION_ITER = int(os.environ.get("ACTIVATION_ITER", "100"))
ACTIVATION_SHOT = int(os.environ.get("ACTIVATION_SHOT", "4"))
REINFORCE_EXAMPLES = int(os.environ.get("REINFORCE_EXAMPLES", "100"))
FLEXIBLE_ACTIVATION_DATASETS = {"SMMILE"}

BENCHMARK_CONFIGS = {
    "mic": [
        ("coco", Path("./datalists/coco/dataset-18.json"), 8),
        ("vqa", Path("./datalists/vqa/dataset-18.json"), 8),
        ("flickr", Path("./datalists/flickr/dataset-18.json"), 8),
        ("miniimage", Path("./datalists/miniimage/dataset-18.json"), 8),
        ("gqa", Path("./datalists/gqa/dataset-18.json"), 8),
        ("nlvr2", Path("./datalists/nlvr2/dataset-18.json"), 8),
        ("iconqa", Path("./datalists/iconqa/dataset-18.json"), 8),
        ("okvqa", Path("./datalists/okvqa/dataset-18.json"), 8),
        ("stvqa", Path("./datalists/stvqa/dataset-18.json"), 8),
        ("textvqa", Path("./datalists/textvqa/dataset-18.json"), 8),
    ],
    "truemicl": [
        ("classification", Path("./datalists/classification/dataset-8_truemicl.json"), 8),
        ("clevr", Path("./datalists/clevr/dataset-8_truemicl.json"), 8),
        ("operator_induction", Path("./datalists/operator_induction/dataset-8_truemicl.json"), 8),
        ("palindrome_dataset", Path("./datalists/palindrome_dataset/dataset-8.json"), 8),
        ("shapes_count", Path("./datalists/shapes_count/dataset-8.json"), 8),
        ("sudoku", Path("./datalists/sudoku/dataset-8.json"), 8),
        ("clock", Path("./datalists/clock/dataset-8.json"), 8),
    ],
    "vlicl": [
        ("clevr", Path("./datalists/clevr/dataset-18.json"), 8),
        ("textocr", Path("./datalists/textocr/dataset-18.json"), 8),
        ("operator_induction", Path("./datalists/operator_induction/dataset-18.json"), 8),
        ("operator_induction_interleaved", Path("./datalists/operator_induction_interleaved/dataset-18.json"), 8),
    ],
    "vliclv2": [
        ("clevr", Path("./datalists/vliclv2/clevr/dataset-8.json"), 8),
        ("textocr", Path("./datalists/vliclv2/operator_induction/dataset-8.json"), 8),
        ("operator_induction", Path("./datalists/vliclv2/operator_induction_interleaved/dataset-8.json"), 8),
        ("operator_induction_interleaved", Path("./datalists/vliclv2/textocr/dataset-8.json"), 8),
    ],
    "isekai": [
        ("isekai", Path("./datalists/isekai/dataset-16.json"), 16),
    ],
    "isekai_4choice": [
        ("isekai_4choice", Path("./datalists/isekai_4choice/dataset-16.json"), 16),
    ],
    "SMMILE": [
        ("SMMILE_choice", Path("./datalists/SMMILE/dataset-4_choice.json"), 4),
        ("SMMILE_openended", Path("./datalists/SMMILE/dataset-4_openended.json"), 4),
    ],
    "illusionvqa-comp": [
        ("illusionvqa-comp", Path("./datalists/illusionvqa-comp/dataset-8.json"), 8),
    ],
    "idk": [
        ("idk", Path("./datalists/idk/dataset-8.json"), 8),
    ],
    "pororov2": [
        ("pororov2", Path("./datalists/pororov2/dataset-4.json"), 4),
    ],
    "pororov3": [
        ("pororov3", Path("./datalists/pororov3/dataset-4.json"), 4),
    ],
    "pororov4": [
        ("pororov4", Path("datalists/pororov4/train/dataset-4.json"), 4),
    ],
    "flintstonesv3": [
        ("flintstonesv3", Path("datalists/flintstonesv3/train/dataset-4.json"), 4),
    ],
    "story_visual_ICLv2": [
        ("story_visual_ICLv2", Path("./datalists/story_visual_ICLv2/dataset-4.json"), 4),
    ],
    "storystreamv2": [
        ("storystreamv2", Path("./datalists/storystreamv2/dataset-4.json"), 4),
    ],
    "pororov4_new": [
        ("pororov4_new", Path("./datalists/pororov4_new/dataset-4.json"), 4),
    ],
    "flintstonesv3_new": [
        ("flintstonesv3_new", Path("./datalists/flintstonesv3_new/dataset-4.json"), 4),
    ],
    "storystreamv2_new": [
        ("storystreamv2_new", Path("./datalists/storystreamv2_new/dataset-4.json"), 4),
    ],
}


def split_prompt_answer(segment: str) -> tuple[str, str]:
    cleaned = segment.strip()
    for marker in ("Answer:", "The answer is "):
        marker_idx = cleaned.rfind(marker)
        if marker_idx != -1:
            prefix = cleaned[: marker_idx + len(marker)]
            answer = cleaned[marker_idx + len(marker) :].strip()
            if not answer:
                raise ValueError(f"Failed to parse answer from segment: {cleaned}")
            if marker == "Answer:":
                prefix = prefix.rstrip() + " "
            return prefix, answer

    question_idx = cleaned.rfind("? ")
    if question_idx != -1:
        prefix = cleaned[: question_idx + 2]
        answer = cleaned[question_idx + 2 :].strip()
        if answer:
            return prefix, answer

    sentence_idx = cleaned.rfind(". ")
    if sentence_idx != -1:
        prefix = cleaned[: sentence_idx + 2]
        answer = cleaned[sentence_idx + 2 :].strip()
        if answer:
            return prefix, answer

    raise ValueError(f"Unsupported support format: {cleaned}")


def extract_support_examples(subtask: str, datalist_path: Path, support_count: int) -> tuple[list[dict], list[dict], int]:
    items = json.loads(datalist_path.read_text(encoding="utf-8"))
    support_examples = []
    support_groups = []
    skipped_examples = 0

    for item in items:
        images = item["image"]
        human_text = item["conversations"][0]["value"]
        parts = human_text.split("<image>")
        intro = parts[0].strip()
        segments = parts[1:]

        example_count = support_count + 1
        if len(images) % example_count != 0:
            raise ValueError(
                f"{subtask}: image count {len(images)} is not divisible by example count {example_count} for id={item['id']}"
            )
        images_per_example = len(images) // example_count
        if len(segments) != len(images):
            raise ValueError(
                f"{subtask}: expected {len(images)} <image> segments, got {len(segments)} for id={item['id']}"
            )

        group_examples = []
        for support_idx in range(support_count):
            segment_start = support_idx * images_per_example
            segment_end = segment_start + images_per_example
            grouped_segment = "".join(segments[segment_start:segment_end])
            try:
                prompt_prefix, answer = split_prompt_answer(grouped_segment)
            except ValueError:
                skipped_examples += 1
                continue
            example = {
                "key": f"{subtask}::{item['id']}::{support_idx}",
                "source_id": item["id"],
                "subtask": subtask,
                "intro": intro,
                "prompt_prefix": prompt_prefix,
                "answer": answer,
                "images": images[segment_start:segment_end],
                "image_token_count": images_per_example,
            }
            support_examples.append(example)
            group_examples.append(example)

        support_groups.append(
            {
                "group_key": f"{subtask}::{item['id']}",
                "subtask": subtask,
                "source_id": item["id"],
                "examples": group_examples,
            }
        )

    return support_examples, support_groups, skipped_examples


def build_activation_text(examples: list[dict]) -> str:
    intro = examples[0]["intro"].strip()
    chunks = []
    if intro:
        chunks.append(intro)
    for example in examples:
        image_tokens = "<image>" * example["image_token_count"]
        chunks.append(f"{image_tokens}{example['prompt_prefix']}{example['answer']}".strip())
    return "\n".join(chunks).strip()


def build_single_text(example: dict) -> str:
    intro = example["intro"].strip()
    image_tokens = "<image>" * example["image_token_count"]
    prompt = f"{image_tokens}{example['prompt_prefix'].rstrip()}".strip()
    if intro:
        return f"{intro}\n{prompt}".strip()
    return prompt


def make_activation_record(iter_idx: int, subtask: str, examples: list[dict]) -> dict:
    images = []
    for example in examples:
        images.extend(example["images"])
    return {
        "id": f"{DATASET_NAME}_activation_iter_{iter_idx}",
        "content": STANDARD_CONTENT,
        "task": subtask,
        "image": images,
        "conversations": [
            {"from": "human", "value": build_activation_text(examples)},
            {"from": "gpt", "value": ""},
        ],
    }


def make_reinforce_record(example: dict, record_idx: int) -> dict:
    return {
        "id": f"{DATASET_NAME}_reinforce_{record_idx}",
        "content": STANDARD_CONTENT,
        "task": example["subtask"],
        "image": example["images"],
        "conversations": [
            {"from": "human", "value": build_single_text(example)},
            {"from": "gpt", "value": example["answer"]},
        ],
    }


if DATASET_NAME not in BENCHMARK_CONFIGS:
    raise ValueError(f"Unknown benchmark: {DATASET_NAME}")

pool_by_subtask = {}
groups_by_subtask = {}
for subtask, datalist_path, support_count in BENCHMARK_CONFIGS[DATASET_NAME]:
    support_examples, support_groups, skipped_examples = extract_support_examples(subtask, datalist_path, support_count)
    if not support_examples:
        raise ValueError(f"{subtask}: empty support pool")
    pool_by_subtask[subtask] = support_examples
    groups_by_subtask[subtask] = support_groups
    if skipped_examples:
        print(f"[MTV]   {subtask}: skipped {skipped_examples} invalid support examples")

subtasks = [subtask for subtask, _, _ in BENCHMARK_CONFIGS[DATASET_NAME]]
activation_records = []
activation_iter_counts = {subtask: 0 for subtask in subtasks}


def balanced_counts(total: int, names: list[str]) -> dict[str, int]:
    base = total // len(names)
    remainder = total % len(names)
    counts = {name: base for name in names}
    for name in names[:remainder]:
        counts[name] += 1
    return counts


reinforce_target_counts = balanced_counts(REINFORCE_EXAMPLES, subtasks)
for subtask in subtasks:
    if len(pool_by_subtask[subtask]) < reinforce_target_counts[subtask]:
        raise ValueError(
            f"{DATASET_NAME}: subtask {subtask} needs {reinforce_target_counts[subtask]} reinforce examples "
            f"but only {len(pool_by_subtask[subtask])} support examples are available"
        )

reinforce_examples = []
used_reinforce_keys = set()
for subtask in subtasks:
    chosen_examples = RNG.sample(pool_by_subtask[subtask], reinforce_target_counts[subtask])
    reinforce_examples.extend(chosen_examples)
    used_reinforce_keys.update(example["key"] for example in chosen_examples)

remaining_groups_by_subtask = {subtask: [] for subtask in subtasks}
remaining_pool_by_subtask = {subtask: [] for subtask in subtasks}
for subtask in subtasks:
    for group in groups_by_subtask[subtask]:
        remaining_examples = [example for example in group["examples"] if example["key"] not in used_reinforce_keys]
        remaining_pool_by_subtask[subtask].extend(remaining_examples)
        if len(remaining_examples) >= ACTIVATION_SHOT:
            remaining_groups_by_subtask[subtask].append(
                {
                    "group_key": group["group_key"],
                    "subtask": subtask,
                    "source_id": group["source_id"],
                    "examples": remaining_examples,
                }
            )

activation_target_counts = balanced_counts(ACTIVATION_ITER, subtasks)
for subtask in subtasks:
    if DATASET_NAME in FLEXIBLE_ACTIVATION_DATASETS:
        if len(remaining_pool_by_subtask[subtask]) < ACTIVATION_SHOT:
            raise ValueError(
                f"{DATASET_NAME}: subtask {subtask} has only {len(remaining_pool_by_subtask[subtask])} "
                f"support examples remaining after reinforce exclusion, need at least {ACTIVATION_SHOT}"
            )
    else:
        if not remaining_groups_by_subtask[subtask]:
            raise ValueError(
                f"{DATASET_NAME}: subtask {subtask} has no source ids with at least {ACTIVATION_SHOT} "
                "support examples after reinforce exclusion"
            )

activation_plan = []
for subtask in subtasks:
    activation_plan.extend([subtask] * activation_target_counts[subtask])
RNG.shuffle(activation_plan)

for iter_idx, subtask in enumerate(activation_plan):
    if DATASET_NAME in FLEXIBLE_ACTIVATION_DATASETS:
        chosen = RNG.sample(remaining_pool_by_subtask[subtask], ACTIVATION_SHOT)
    else:
        group = RNG.choice(remaining_groups_by_subtask[subtask])
        chosen = RNG.sample(group["examples"], ACTIVATION_SHOT)
    activation_iter_counts[subtask] += 1
    activation_records.append(make_activation_record(iter_idx, subtask, chosen))

RNG.shuffle(reinforce_examples)
reinforce_records = [make_reinforce_record(example, idx) for idx, example in enumerate(reinforce_examples)]

ACTIVATION_JSON.write_text(json.dumps(activation_records, ensure_ascii=False, indent=2), encoding="utf-8")
REINFORCE_JSON.write_text(json.dumps(reinforce_records, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[MTV] Wrote {DATASET_NAME} activation JSON: {ACTIVATION_JSON}")
print(f"[MTV] Wrote {DATASET_NAME} reinforce JSON: {REINFORCE_JSON}")
print(f"[MTV] Activation iters: {len(activation_records)}")
print(f"[MTV] Reinforce singles: {len(reinforce_records)}")
print(f"[MTV] Unique reinforce support examples: {len(used_reinforce_keys)}")
for subtask in subtasks:
    print(
        f"[MTV]   {subtask}: {len(pool_by_subtask[subtask])} support examples, "
        f"{len(remaining_groups_by_subtask[subtask])} eligible activation groups, "
        f"{len(remaining_pool_by_subtask[subtask])} remaining support examples, "
        f"{activation_iter_counts[subtask]} activation iters, "
        f"{reinforce_target_counts[subtask]} reinforce singles"
    )
PY

    if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
        return 0
    fi

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -m MTV.mtv_eval2 \
        --model_name "${MODEL}" \
        --data_name "${dataset_name}" \
        --train_path "${activation_json}" \
        --val_path "${reinforce_json}" \
        --reinforce_path "${reinforce_json}" \
        --num_example "${ACTIVATION_ITER}" \
        --num_shot "${ACTIVATION_SHOT}" \
        --max_token 512 \
        --eval_num_shot 0 \
        --bernoullis_path "${storage_dir}/mtv_${dataset_name}.pt" \
        --activation_path "${storage_dir}/mtv_activation_${dataset_name}.pt" \
        --is_eval True \
        --result_folder ./ \
        --cur_mode both \
        --experiment_name "${dataset_name}"
}

BENCHMARKS=(mic truemicl vlicl vliclv2 isekai isekai_4choice SMMILE illusionvqa-comp idk pororov2 pororov3 pororov4 flintstonesv3 story_visual_ICLv2 storystreamv2 pororov4_new flintstonesv3_new storystreamv2_new)

FAILURES=()

for MODEL in "${MODELS[@]}"; do
    echo "[MTV] ===== Model: ${MODEL} (GPU ${GPU_ID}) ====="
    for dataset_name in "${BENCHMARKS[@]}"; do
        if [[ "${BENCHMARK_FILTER}" != "all" && "${BENCHMARK_FILTER}" != "${dataset_name}" ]]; then
            continue
        fi
        # Keep going through the remaining models when one run dies, instead of
        # letting `set -e` abort the whole sweep.
        if ! run_benchmark "${dataset_name}"; then
            echo "[MTV] FAILED: ${MODEL} / ${dataset_name}" >&2
            FAILURES+=("${MODEL}/${dataset_name}")
        fi
    done
done

if (( ${#FAILURES[@]} > 0 )); then
    echo "[MTV] ${#FAILURES[@]} run(s) failed: ${FAILURES[*]}" >&2
    exit 1
fi

echo "[MTV] All runs finished: ${MODELS[*]}"
