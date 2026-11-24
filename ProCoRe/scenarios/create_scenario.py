import random
import os
import argparse
import json

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()
    
    scenarios = [1,2,3,4,5,6,7,8,9,10]
    datasets = ["mic", "truemicl"]
    sub_datasets = ["coco", "flickr", "miniimage", "gqa", "nlvr2", "iconqa", "okvqa", "stvqa", "textvqa", "vqav2"]
    ordering = "lure"
    models = ["idefics-8b", "onevision1_5-8b", "qwen3-8b", "internvl3_5-8b"]
    shots = [0,1,2,4,8]
    
    # random.shuffle(task_list)
    for shot_num in shots:
        for i in range(len(models)):
            for dataset in datasets:
                if dataset == "mic":
                    sub_datasets = ["coco", "flickr", "miniimage", "gqa", "nlvr2", "iconqa", "okvqa", "stvqa", "textvqa", "vqav2"]
                elif dataset == "truemicl":
                    sub_datasets = ["classification", "clevr", "operator_induction", "plaindrome", "shapescount", "sudoku", "clock"]
                else:
                    subdatasets = [dataset]
                model = models[i]
                
                for j, subdataset in enumerate(sub_datasets):
                    scenario_data = []
                    if (dataset=="mic" and subdataset=="nlvr2") or (dataset=="truemicl" and subdataset=="classification") or dataset=="isekai":
                        scenario_data.append({
                            'client_id':0,
                            'datasets':[{"dataset": f"{subdataset}", "subset_id": f"from_{model}_scen{j+1}_{shot_num}shot"}],
                            "type": "exact-match", "item": "accuracy"
                        })
                    elif dataset=="truemicl":
                        scenario_data.append({
                            'client_id':0,
                            'datasets':[{"dataset": f"{subdataset}", "subset_id": f"from_{model}_scen{j+1}_{shot_num}shot"}],
                            "type": "multi-choice", "item": "accuracy"
                        })
                    else:
                        scenario_data.append({
                            'client_id':0,
                            'datasets':[{"dataset": f"{subdataset}", "subset_id": f"from_{model}_scen{j+1}_{shot_num}shot"}],
                            "type": "open-ended", "item": "ROUGE-L"
                        })
                    
                    os.makedirs(f"../scenarios/{shot_num}-shot/{dataset}/{ordering}", exist_ok=True)
                    print(j)
                    output_path = f'../scenarios/{shot_num}-shot/{dataset}/{ordering}/scenario-{(i+1)*(j+1)}.json'
                    print(output_path)
                    with open(output_path, 'w') as json_file:
                        json.dump(scenario_data, json_file, indent=4)
    
if __name__ == "__main__":
    main()