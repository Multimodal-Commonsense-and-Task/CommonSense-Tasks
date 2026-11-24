set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

MODEL="${MODEL:-onevision1_5-8b}" # onevision1_5-8b, qwen3-8b, internvl3_5-8b, idefics-8b, otter-9b
DATASET_NAME="truemicl"
ACTIVATION_ITER=100
ACTIVATION_SHOT=4
REINFORCE_EXAMPLES=100

STORAGE_DIR="MTV/storage/${MODEL}/${DATASET_NAME}"
STORAGE_JSON_DIR="MTV/storage_json/${MODEL}/${DATASET_NAME}"
mkdir -p "${STORAGE_DIR}" "${STORAGE_JSON_DIR}"

TRAIN_MIX_PATH="${STORAGE_JSON_DIR}/mtv_activation_only_truemicl.json"
REINFORCE_MIX_PATH="${STORAGE_JSON_DIR}/mtv_reinforce_only_truemicl.json"

python3 - <<'PY' "${TRAIN_MIX_PATH}" "${REINFORCE_MIX_PATH}" "${REPO_ROOT}"
import json
import random
import sys
from pathlib import Path

activation_out = Path(sys.argv[1])
reinforce_out = Path(sys.argv[2])
repo_root = Path(sys.argv[3])

rng = random.Random(0)
data_root = repo_root / "data" / "truemicl"

seed_counts = {
    "classification": 13,
    "clock": 13,
    "operator_induction": 13,
    "palindrome_dataset": 26,
    "shapes_count": 26,
    "sudoku": 13,
    "clevr": 120,
}

reinforce_counts = {
    "classification": 3,
    "clock": 3,
    "operator_induction": 3,
    "palindrome_dataset": 6,
    "shapes_count": 6,
    "sudoku": 3,
    "clevr": 76,
}

iter_counts = {
    "classification": 6,
    "clock": 6,
    "operator_induction": 6,
    "palindrome_dataset": 11,
    "shapes_count": 11,
    "sudoku": 6,
    "clevr": 54,
}

def as_text(answer):
    return str(answer)

def resolve_answer(task_name, answer, answer_idx=None):
    if task_name in {"clock", "operator_induction"}:
        if answer_idx is None:
            raise ValueError(f"answer_idx is required for {task_name}")
        return answer[answer_idx]
    return answer

def build_single_prompt(task_name, question):
    question = question.strip()
    if task_name == "classification":
        return f"\n<image>\n{question}\nGive only the answer \nAnswer:"
    if task_name == "clevr":
        return "Look carefully at the image, count ONLY the objects that match the given property, and output ONLY the number.\n<image>\nHow many objects match this property: {}? Give only the answer \nAnswer:".format(question)
    return f"\n<image>\n{question} Give only the answer \nAnswer:"

def load_support(task_name):
    filename = "support.json"
    path = data_root / task_name / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_single_record(task_name, item, answer_idx=None):
    return {
        "id": item["id"],
        "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        "task": task_name,
        "image": [f"./data/{item['image'][0]}"],
        "conversations": [
            {"from": "human", "value": build_single_prompt(task_name, item["question"])},
            {"from": "gpt", "value": as_text(resolve_answer(task_name, item["answer"], answer_idx))},
        ],
    }

def make_many_shot_record(task_name, iter_idx, samples, answer_idx=None):
    prompt_parts = []
    images = []
    for sample in samples:
        answer = resolve_answer(task_name, sample["answer"], answer_idx)
        prompt_parts.append(build_single_prompt(task_name, sample["question"]) + f" {as_text(answer)}\n")
        images.append(f"./data/{sample['image'][0]}")
    return {
        "id": f"{task_name}_iter_{iter_idx}",
        "content": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        "task": task_name,
        "image": images,
        "conversations": [
            {"from": "human", "value": "".join(prompt_parts).strip()},
            {"from": "gpt", "value": ""},
        ],
    }

task_pools = {}
for task_name in seed_counts:
    task_pools[task_name] = load_support(task_name)

reinforce_records = []
seed_pools = {}

for task_name, all_samples in task_pools.items():
    if len(all_samples) < seed_counts[task_name] + reinforce_counts[task_name]:
        raise ValueError(f"{task_name} does not have enough support samples")

    shuffled = all_samples[:]
    rng.shuffle(shuffled)
    reinforce_samples = shuffled[:reinforce_counts[task_name]]
    seed_samples = shuffled[reinforce_counts[task_name]:reinforce_counts[task_name] + seed_counts[task_name]]

    for sample in reinforce_samples:
        answer_idx = rng.randrange(3) if task_name in {"clock", "operator_induction"} else None
        reinforce_records.append(make_single_record(task_name, sample, answer_idx=answer_idx))
    seed_pools[task_name] = seed_samples

def sample_episode(seed_samples, shot, local_rng):
    if len(seed_samples) >= shot:
        episode = local_rng.sample(seed_samples, shot)
    else:
        episode = [local_rng.choice(seed_samples) for _ in range(shot)]
    local_rng.shuffle(episode)
    return episode

activation_records = []
for task_name, num_iters in iter_counts.items():
    for idx in range(num_iters):
        samples = sample_episode(seed_pools[task_name], 4, rng)
        answer_idx = rng.randrange(3) if task_name in {"clock", "operator_induction"} else None
        activation_records.append(make_many_shot_record(task_name, idx, samples, answer_idx=answer_idx))

rng.shuffle(activation_records)
rng.shuffle(reinforce_records)

if len(activation_records) != 100:
    raise ValueError(f"Expected 100 activation records, got {len(activation_records)}")
if len(reinforce_records) != 100:
    raise ValueError(f"Expected 100 reinforce records, got {len(reinforce_records)}")

activation_out.write_text(json.dumps(activation_records, ensure_ascii=False), encoding="utf-8")
reinforce_out.write_text(json.dumps(reinforce_records, ensure_ascii=False), encoding="utf-8")

print("[MTV] Wrote TrueMICL activation/reinforce datalists")
print("[MTV] Activation iter counts:")
for task_name in iter_counts:
    print(f"  - {task_name}: {iter_counts[task_name]} iter ({iter_counts[task_name] * 4} slots)")
print("[MTV] Reinforce counts:")
for task_name in reinforce_counts:
    print(f"  - {task_name}: {reinforce_counts[task_name]}")
print("[MTV] Seed counts:")
for task_name in seed_counts:
    print(f"  - {task_name}: {seed_counts[task_name]}")
PY

if [[ "${GENERATE_ONLY:-0}" == "1" ]]; then
    exit 0
fi

python3 -m MTV.mtv_eval2 \
    --model_name "${MODEL}" \
    --data_name "${DATASET_NAME}" \
    --train_path "${TRAIN_MIX_PATH}" \
    --val_path "${REINFORCE_MIX_PATH}" \
    --reinforce_path "${REINFORCE_MIX_PATH}" \
    --num_example "${ACTIVATION_ITER}" \
    --num_shot "${ACTIVATION_SHOT}" \
    --max_token 512 \
    --eval_num_shot 0 \
    --bernoullis_path "${STORAGE_DIR}/mtv_only_truemicl.pt" \
    --activation_path "${STORAGE_DIR}/mtv_activation_only_truemicl.pt" \
    --is_eval True \
    --result_folder ./ \
    --cur_mode both \
    --experiment_name "${DATASET_NAME}"
