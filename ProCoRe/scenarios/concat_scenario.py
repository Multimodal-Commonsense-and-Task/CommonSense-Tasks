import os
import json

BASE_DIR = "/disk1/mjlee/LMM-ICL/scenarios"
OUTPUT_NAME = "scenario-0.json"

for root, dirs, files in os.walk(BASE_DIR, topdown=True):
    # Only operate on leaf (final) folders
    print("here", dirs)
    if dirs:
        continue

    # Collect JSON files in this leaf, excluding our output file
    json_files = [f for f in files if f.endswith(".json") and f != OUTPUT_NAME]
    
    if not json_files:
        continue

    combined_datasets = []

    for jf in sorted(json_files):
        path = os.path.join(root, jf)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON, skipped: {path}")
            continue
        except Exception as e:
            print(f"[WARN] Error reading {path}: {e}")
            continue

        # Expected format per file:
        # [
        #   {"client_id": <int>, "datasets": [ ... list of dataset dicts ... ]},
        #   ... (possibly more items)
        # ]
        if not isinstance(data, list):
            print(f"[WARN] Top-level JSON is not a list, skipped: {path}")
            continue

        for entry in data:
            if isinstance(entry, dict) and "datasets" in entry and isinstance(entry["datasets"], list):
                combined_datasets.extend(entry["datasets"])
            else:
                print(f"[WARN] Entry missing 'datasets' list, skipped: {path}")

    # Build the output in the SAME format
    output_obj = [{"client_id": 0, "datasets": combined_datasets}]

    out_path = os.path.join(root, OUTPUT_NAME)
    try:
        with open(out_path, "w") as out:
            json.dump(output_obj, out, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {out_path} (datasets: {len(combined_datasets)})")
    except Exception as e:
        print(f"[ERR] Could not write {out_path}: {e}")
