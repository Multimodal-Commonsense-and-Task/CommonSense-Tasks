import torch
from utils.train_utils import load_deepspeed, get_target_layers
import copy

def fedmosaic_homo_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict=None):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)  
            
        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # active clients
            active_clients = extra_state_dict_dict['selected_ids']
            
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            new_global_state_dict = {}
            
            weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/training_args.softmax_temp).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                else:
                    new_global_state_dict[name] = local_state_dict_list[client_id][name]
                    continue
                
                for id in active_clients:
                    if id == client_id:
                        continue
                    # if training_args.is_hetero_model:
                    #     breakpoint()
                    # else:
                    new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
                if isinstance(new_param, int):
                    continue
                new_global_state_dict[name] = new_param
        else:
            new_global_state_dict = global_state_dict
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 


def fedmosaic_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    # first load loca model and then load global model
    layer_index = extra_state_dict_dict['LAYER_INDEX']
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)
            
        model_ids = extra_state_dict_dict['model_ids']
        
        for model_id, homo_ids in model_ids.items():
            if client_id in homo_ids:
                homo_client_ids = homo_ids    
        
        new_global_state_dict = {}
        for key in local_state_dict_list[client_id].keys():
            if 'lora2' in key:
                new_key = key.replace('lora2','lora1')
                new_global_state_dict[new_key] = copy.deepcopy(local_state_dict_list[client_id][key])

        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            
            weights = sim[client_id].clone()
            homo_weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/training_args.softmax_temp).softmax(dim=0)
            sim_sum = weights.sum() - weights[client_id]
            
            for id in range(training_args.num_clients):
                if id not in homo_client_ids:
                    homo_weights[id] = -1e9
            
            homo_weights[client_id] = -1e9
            homo_weights = (homo_weights/training_args.softmax_temp).softmax(dim=0)
            
            homo_sim_sum = homo_weights.sum() - homo_weights[client_id]
            
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            cur_layer_num = []
            for k in global_state_dict.keys():
                if 'layers.' in k:
                    cur_layer_num.append(int(k.split('.')[layer_index]))
            cur_layer_num = sorted(list(set(cur_layer_num)))
            cur_layer_num = get_target_layers(len(set(cur_layer_num)), training_args.num_blocks)
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                else:
                    new_global_state_dict[name] = local_state_dict_list[client_id][name]
                    continue
                
                splited = target_key.split('.')
                if int(splited[layer_index]) in cur_layer_num:
                    if 'lora2_P' not in target_key and 'lora2_Q' not in target_key:
                        continue
                    
                    for id in range(training_args.num_clients):
                        if id == client_id:
                            continue
                        else:
                            splited = target_key.split('.')
                            # if layer number is different
                            layer_num = []
                            for k in local_state_dict_list[id].keys():
                                if 'layers.' in k:
                                    layer_num.append(int(k.split('.')[layer_index]))
                            layer_num = len(set(layer_num))
                            target_layers = get_target_layers(layer_num, training_args.num_blocks)
                            if cur_layer_num[-1] != target_layers[-1]: # if different size
                                index = cur_layer_num.index(int(splited[layer_index]))
                                splited[layer_index] = str(target_layers[index])
                                new_target_key = '.'.join(splited)
                            else:
                                new_target_key = target_key
                            new_param += weights[id]*local_state_dict_list[id][new_target_key] / sim_sum
                else:
                    for id in homo_client_ids:
                        if id == client_id:
                            continue
                        new_param += homo_weights[id]*local_state_dict_list[id][target_key] / homo_sim_sum
                if isinstance(new_param, int):
                    continue
                new_global_state_dict[name] = new_param

        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 
