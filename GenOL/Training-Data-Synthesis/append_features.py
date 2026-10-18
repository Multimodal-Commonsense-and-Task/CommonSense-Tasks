import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--feature_path", type=str, required=True)
args = parser.parse_args()

def main():
    # Load jsonl file
    with open(os.path.join(args.image_dir, "metadata.jsonl"), 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    
    # Append features to jsonl file
    feature_classes = os.listdir(args.feature_path)
    for i, row in enumerate(data):
        file_name = row["file_name"]
        image_extension = file_name.split(".")[-1]
        class_name = file_name.split("/")[0]
        image_name = file_name.replace(f".{image_extension}", ".pt")
        feature_file_name = os.path.join(args.feature_path, image_name)

        if not os.path.exists(feature_file_name):
            raise FileNotFoundError(f"Feature file {feature_file_name} not found.")

        data[i]["img_features"] = feature_file_name
        data[i]["class"] = class_name

    with open(os.path.join(args.image_dir, "metadata.jsonl"), 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')
    
    
if __name__ == "__main__":
    main()