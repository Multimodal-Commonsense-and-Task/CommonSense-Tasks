import pickle
import json
from pathlib import Path

pickle_path = './RMD_scores/PACS_final_sd3.pkl'
json_path = './RMD_scores/PACS_final_sd3.json'

with open(pickle_path, 'rb') as f:
    RMD_scores = pickle.load(f)
    
result_dict = {}
for (model_name, class_name), samples_list in RMD_scores.items():
    if model_name not in result_dict:
        result_dict[model_name] = {}
    if class_name not in result_dict[model_name]:
        result_dict[model_name][class_name] = []
    
    for sample_path, score in samples_list:
        path = Path(sample_path)
        path = str(Path(*path.parts[-3:]))
        result_dict[model_name][class_name].append({
            'image_path': path,
            'score': score
        })
        
with open(json_path, 'w') as f:
    json.dump(result_dict, f)