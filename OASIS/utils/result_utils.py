import csv
import json
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple


ScenarioItem = Dict[str, object]
ScoreReader = Callable[[ScenarioItem, int], float]


def load_scenario(scenario_num: int) -> List[ScenarioItem]:
    with open(f"./scenarios/scenario-{scenario_num}.json", "r") as fp:
        return json.load(fp)


def result_prefix(is_client: bool, client_id: int, num_rounds: int) -> str:
    role = f"client{client_id}" if is_client else "server"
    return f"{role}_round{num_rounds}"


def read_eval_json(path: str) -> Dict[str, object]:
    with open(path, "r") as fp:
        return json.load(fp)[-1]


def score_from_eval_result(result: Dict[str, object], data: ScenarioItem) -> float:
    if data["type"] == "multi-choice":
        return result["accuracy"]

    if data["metric"] == "F1":
        precision = result["precision"]
        recall = result["recall"]
        return 2 * (precision * recall) / (precision + recall)

    if data["metric"] == "RougeL":
        return result["ROUGE_L"][0]

    raise ValueError(f"Unsupported open-ended metric: {data['metric']}")


def read_gpt_score(path: str) -> float:
    import jsonlines

    count = 0
    total_score = 0
    with jsonlines.open(path) as read_file:
        for line in read_file.iter():
            total_score += line["score"]
            count += 1
    return total_score / count / 10


def collect_scores(
    scenario: Iterable[Dict[str, object]],
    read_score: ScoreReader,
) -> Tuple[Dict[str, float], Dict[int, float]]:
    scores = {}
    client_scores = defaultdict(list)

    for client_data in scenario:
        client_id = client_data["client_id"]
        for data in client_data["datasets"]:
            data_name = f"{data['dataset']}-{data['subset_id']}"
            score = read_score(data, client_id)
            scores[data_name] = score
            client_scores[client_id].append(score)

    client_avg_scores = {
        client_id: sum(values) / len(values)
        for client_id, values in client_scores.items()
    }
    return scores, client_avg_scores


def append_results_csv(
    method: str,
    scores: Dict[str, float],
    client_avg_scores: Dict[int, float],
    csv_file: str = "results.csv",
    num_clients: int = 10,
) -> float:
    avg_score = sum(scores.values()) / len(scores)
    header = ["method", "final score"]
    header.extend(f"client {client_id}" for client_id in range(num_clients))
    header.extend(scores.keys())

    csv_data = [method, avg_score]
    csv_data.extend(client_avg_scores.get(client_id, "") for client_id in range(num_clients))
    csv_data.extend(scores.values())

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(csv_data)

    return avg_score


def report_results(method: str, scores: Dict[str, float], avg_score: float, csv_file: str) -> None:
    print(f"Method: {method}")
    print(f"Score: {scores}")
    print(f"Avg Score: {avg_score}")
    print(f"Results have been appended to {csv_file}")
