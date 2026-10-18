# This script removes the absolute path in RMD pickle file
import pickle
import os

PICKLE_PATH = './RMD_scores/RMD_scores_PACS_web.pkl'

with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

modified_dict = {}
for k, v in data.items():
    values_list = []
    for path, score in v:
        filename = os.path.basename(path)
        directory = os.path.basename(os.path.dirname(path))
        specific_path = os.path.join('.', directory, filename)
        values_list.append((specific_path, score))
    
    modified_dict[k] = values_list

breakpoint()