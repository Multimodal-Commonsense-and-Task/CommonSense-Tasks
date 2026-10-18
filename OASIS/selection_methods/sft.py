import torch
from utils.train_utils import load_deepspeed

def sft_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict):
    model_to_load = local_state_dict_list[client_id]
    with torch.no_grad():
        # if 'zero3' in training_args.deepspeed:
        #     load_deepspeed(model_to_load, model, strict=False)
        # else:
        model.load_state_dict(model_to_load, strict=False)  

