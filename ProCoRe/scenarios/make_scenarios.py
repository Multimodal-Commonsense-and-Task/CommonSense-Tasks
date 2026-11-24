import os
import json

dataset_name = "truemicl"
# sort_methods = ["textsim_incre", "bothsim_incre", "bothsim_decre", "textsim_decre"]
# shots = [1,2,4,8,16]
sort_methods = ["random"]
modality_order = "text"
shots = [0]

if dataset_name == "vlicl":
    eval_methods = ["exact-match","exact-match","exact-match","exact-match"] 
    metric = ["accuracy", "accuracy", "accuracy", "accuracy"]
    datasets = ["clevr", "textocr", "operator_induction", "operator_induction_interleaved"]
elif dataset_name == "mic":
    eval_method = "open-ended"
    metric = "RougeL"
    datasets = ["coco", "flickr", "miniimage", "gqa","nlvr2",  "iconqa", "okvqa" , "stvqa", "textvqa","vqa"]
elif dataset_name == "truemicl":
    eval_methods = ["multi-choice", "exact-match", "exact-match", "exact-match", "exact-match", "exact-match", "exact-match"]
    metric = ["accuracy", "accuracy", "accuracy", "accuracy","accuracy","accuracy","RougeL","accuracy"]
    datasets = ["classification", "clevr","operator_induction", "palindrome_dataset", "shapes_count", "sudoku", "clock"]


for shot in shots:
    for sort_method in sort_methods:
        if sort_method == "random":
            subset_id = int(f"{str(shot)}")
        elif sort_method == "sim_incre":
            subset_id = int(str(f"1{shot}"))
            if shot in [2,4,8]:
                subset_id = f"{shot}_imgsim_asc"
        elif sort_method == "sim_decre":
            subset_id = int(str(f"1{shot}"))
            if shot in [2,4,8]:
                subset_id = f"{shot}_imgsim_des"
        elif sort_method == "bothsim_incre":
            subset_id = f"{str(shot)}_bothsim_asc"
        elif sort_method == "textsim_incre":
            subset_id = f"{str(shot)}_textsim_asc"
        elif sort_method == "bothsim_decre":
            subset_id = f"{str(shot)}_bothsim_des"
        elif sort_method == "textsim_decre":
            subset_id = f"{str(shot)}_textsim_des"
        elif sort_method == "im_sim_incre":
            subset_id = int(f"{str(shot)}")
        elif sort_method == "text_sim_incre":
            subset_id = int(f"3{str(shot)}")
        for d_i, dataset in enumerate(datasets):
            if dataset_name == "truemicl":
                if dataset in ["classification", "clevr", "operator_induction"]:
                    subset_tail = "_truemicl"
                else:
                    subset_tail = ""
                if modality_order == "image":
                    if sort_method == "random":
                        subset_id = f"{str(shot)}{subset_tail}"
                    elif sort_method == "sim_incre":
                        subset_id = f"{str(shot)}{subset_tail}_imgsim_asc"
                    elif sort_method == "bothsim_incre":
                        subset_id = f"{str(shot)}{subset_tail}_bothsim_asc"
                    elif sort_method == "textsim_incre":
                        subset_id = f"{str(shot)}{subset_tail}_textsim_asc"
                    elif sort_method == "bothsim_decre":
                        subset_id = f"{str(shot)}{subset_tail}_bothsim_des"
                    elif sort_method == "textsim_decre":
                        subset_id = f"{str(shot)}{subset_tail}_textsim_des"
                else:
                    if sort_method == "random":
                        subset_id = f"{str(shot)}{subset_tail}_textfirst"
                    elif sort_method == "sim_incre":
                        subset_id = f"{str(shot)}{subset_tail}_imgsim_asc_textfirst"
                    elif sort_method == "bothsim_incre":
                        subset_id = f"{str(shot)}{subset_tail}_bothsim_asc_textfirst"
                    elif sort_method == "textsim_incre":
                        subset_id = f"{str(shot)}{subset_tail}_textsim_asc_textfirst"
                    elif sort_method == "bothsim_decre":
                        subset_id = f"{str(shot)}{subset_tail}_bothsim_des_textfirst"
                    elif sort_method == "textsim_decre":
                        subset_id = f"{str(shot)}{subset_tail}_textsim_des_textfirst"
                
            client_data = []
            
            items = {}
            items["client_id"] = 0
            
            dataset_list = []
            dataset_item = {}
            dataset_item["dataset"] = dataset
            dataset_item["subset_id"] = subset_id
            dataset_item["type"] = eval_methods[d_i]
            dataset_item["item"] = metric[d_i]
            dataset_list.append(dataset_item)
            
            items["datasets"] = dataset_list
            client_data.append(items)
            
            if modality_order == "image":
                os.makedirs(f"./{shot}-shot/{dataset_name}/{sort_method}", exist_ok=True)
                with open(f"./{shot}-shot/{dataset_name}/{sort_method}/scenario-{d_i+1}.json", "w") as w_json:
                    json.dump(client_data, w_json) 
            else:
                os.makedirs(f"./{shot}-shot/{dataset_name}/{sort_method}_textfirst", exist_ok=True)
                with open(f"./{shot}-shot/{dataset_name}/{sort_method}_textfirst/scenario-{d_i+1}.json", "w") as w_json:
                    json.dump(client_data, w_json) 