import json 

def check_json_keys(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("Keys in JSON file:", len(list(data.keys())))

json_file = "Bongard_HOI_descriptors.json"
check_json_keys(json_file)
