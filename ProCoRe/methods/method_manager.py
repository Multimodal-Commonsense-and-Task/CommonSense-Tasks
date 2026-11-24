from typing import Callable, Tuple, Type, Dict

from methods.sft import sft_org_create_trainer, fedavg_aggregate_state_dict, fedavg_load_state_dict
from methods.dara import dara_create_trainer
from methods.only import only_create_trainer
from methods.mimic import mimic_create_trainer
from methods.fv import fv_create_trainer
from methods.stv import stv_create_trainer

def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    extra_modules = {}
    if mode == "sft":
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, sft_org_create_trainer, fedavg_aggregate_state_dict
    elif mode == "grpo":
        from methods.grpo import grpo_create_trainer
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, grpo_create_trainer, fedavg_aggregate_state_dict
    elif mode == "dara":
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, dara_create_trainer, fedavg_aggregate_state_dict
    elif mode == "only":
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, only_create_trainer, fedavg_aggregate_state_dict
    elif mode == "mimic":
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, mimic_create_trainer, fedavg_aggregate_state_dict
    elif mode == "fv":
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fv_create_trainer, fedavg_aggregate_state_dict
    elif mode == "stv":
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, stv_create_trainer, fedavg_aggregate_state_dict
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
