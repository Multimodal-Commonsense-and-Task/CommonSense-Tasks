"""
Compute accuracy for Step 2 evaluation outputs.

Reads a concatenated JSONL file from Step 2, evaluates each sample,
adds a 'correct' field (1 or 0) to every line, and appends a summary
line with total accuracy at the end.

For Task 1/2 (action_summary, action_summary_between), additional metrics:
  - EM: Exact Match (all elements in order)
  - PM: Partial Match (position-wise match ratio)
  - LM: LCS Match (longest common subsequence / answer length)
  - OM: Orderless Match (set intersection / answer length)

Usage:
    python eval/compute_accuracy.py <JSONL_PATH> <QUESTION_TYPE>

Example:
    python eval/compute_accuracy.py results/.../step2/task1_L1.jsonl action_summary
"""

import ast
import json
import re
import sys


def parse_list_output(output):
    """Parse a Python list from model output string."""
    output = output.strip()
    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass
    match = re.search(r"\[([^\]]+)\]", output)
    if match:
        try:
            parsed = ast.literal_eval("[" + match.group(1) + "]")
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
    return None


def parse_int_list_output(output):
    """Parse a list of integers from model output string."""
    output = output.strip()
    try:
        parsed = ast.literal_eval(output)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        if isinstance(parsed, int):
            return [parsed]
    except Exception:
        pass
    match = re.search(r"\[([^\]]+)\]", output)
    if match:
        nums = re.findall(r"\d+", match.group(1))
        if nums:
            return [int(x) for x in nums]
    nums = re.findall(r"\d+", output)
    if nums:
        return [int(x) for x in nums]
    return None


def parse_int_output(output):
    """Parse a single integer from model output string."""
    output = output.strip()
    try:
        return int(output)
    except Exception:
        pass
    nums = re.findall(r"\d+", output)
    if nums:
        return int(nums[0])
    return None


# ==================== Sequence Metrics (Task 1, 2) ====================

def partial_match(pred, gt):
    """PM: position-wise match count / len(gt)."""
    if not gt:
        return 1.0 if not pred else 0.0
    matches = sum(1 for i in range(min(len(pred), len(gt))) if pred[i] == gt[i])
    return matches / len(gt)


def lcs_length(pred, gt):
    """Compute length of longest common subsequence."""
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def lcs_match(pred, gt):
    """LM: LCS length / len(gt)."""
    if not gt:
        return 1.0 if not pred else 0.0
    return lcs_length(pred, gt) / len(gt)


def orderless_match(pred, gt):
    """OM: count of gt elements found in pred / len(gt)."""
    if not gt:
        return 1.0 if not pred else 0.0
    pred_set = set(pred)
    return sum(1 for x in gt if x in pred_set) / len(gt)


# ==================== Sample Evaluation ====================

def evaluate_sample(pred_text, gt_answer, question_type):
    """Compare model prediction with ground truth.

    Returns True if the prediction matches the ground truth.
    """
    if question_type in ("action_summary", "action_summary_between"):
        pred = parse_list_output(pred_text)
        if pred is None:
            return False
        pred = [x.strip().upper() for x in pred]
        gt = [str(x).strip().upper() for x in gt_answer]
        return pred == gt

    elif question_type.startswith("action_niah"):
        pred = parse_int_list_output(pred_text)
        if pred is None:
            return False
        gt = [int(x) for x in gt_answer] if isinstance(gt_answer, list) else [int(gt_answer)]
        return pred == gt

    elif question_type.startswith("action_anomaly"):
        pred = parse_int_output(pred_text)
        if pred is None:
            return False
        return pred == int(gt_answer)

    else:
        return pred_text.strip() == str(gt_answer).strip()


def compute_sequence_metrics(pred_text, gt_answer):
    """Compute PM, LM, OM for a single sample. Returns (pm, lm, om)."""
    pred = parse_list_output(pred_text)
    gt = [str(x).strip().upper() for x in gt_answer]
    if pred is None:
        return 0.0, 0.0, 0.0
    pred = [x.strip().upper() for x in pred]
    return partial_match(pred, gt), lcs_match(pred, gt), orderless_match(pred, gt)


# ==================== Main ====================

def compute_accuracy(jsonl_path, question_type):
    """Compute accuracy, add 'correct' field to each sample, and append summary."""
    with open(jsonl_path, "r") as f:
        all_lines = [json.loads(line) for line in f if line.strip()]
    # Filter out summary lines from previous runs
    samples = [s for s in all_lines if "EM" not in s and "total_acc" not in s]

    total = len(samples)
    correct = 0
    errors = []
    is_seq_task = question_type in ("action_summary", "action_summary_between")
    pm_sum, lm_sum, om_sum = 0.0, 0.0, 0.0

    for sample in samples:
        pred_text = sample.get("model_prediction", {}).get("message", "")
        gt_answer = sample.get("answer")

        if gt_answer is None:
            sample["correct"] = 0
            continue

        is_correct = evaluate_sample(pred_text, gt_answer, question_type)
        sample["correct"] = 1 if is_correct else 0

        if is_correct:
            correct += 1
        else:
            errors.append({
                "id": sample.get("id", "?"),
                "pred": pred_text,
                "gt": gt_answer,
            })

        if is_seq_task:
            pm, lm, om = compute_sequence_metrics(pred_text, gt_answer)
            pm_sum += pm
            lm_sum += lm
            om_sum += om

    acc = correct / total * 100 if total > 0 else 0.0

    # Build summary
    summary = {
        "EM": f"{acc:.1f}%",
        "correct": correct,
        "total": total,
    }
    if is_seq_task and total > 0:
        summary["PM"] = f"{pm_sum / total * 100:.1f}%"
        summary["LM"] = f"{lm_sum / total * 100:.1f}%"
        summary["OM"] = f"{om_sum / total * 100:.1f}%"

    # Rewrite file with correct field + summary line
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
        f.write(json.dumps(summary) + "\n")

    print(f"  EM: {correct}/{total} = {acc:.1f}%")
    if is_seq_task and total > 0:
        print(f"  PM: {pm_sum / total * 100:.1f}%")
        print(f"  LM: {lm_sum / total * 100:.1f}%")
        print(f"  OM: {om_sum / total * 100:.1f}%")

    # Show a few error examples
    if errors and len(errors) <= 5:
        for e in errors[:3]:
            print(f"    [WRONG] id={e['id']}, pred='{e['pred']}', gt={e['gt']}")
    elif errors:
        for e in errors[:3]:
            print(f"    [WRONG] id={e['id']}, pred='{e['pred']}', gt={e['gt']}")
        print(f"    ... and {len(errors) - 3} more errors")

    return acc


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval/compute_accuracy.py <JSONL_PATH> <QUESTION_TYPE>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    question_type = sys.argv[2]

    print(f"  File: {jsonl_path}")
    print(f"  Task: {question_type}")
    compute_accuracy(jsonl_path, question_type)
