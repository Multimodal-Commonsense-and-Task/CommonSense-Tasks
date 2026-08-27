from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer 
from federated_methods.sft import sft_load_state_dict, fedper_load_state_dict
from federated_methods.fedmosaic import fedmosaic_create_trainer, fedmosaic_aggregate_state_dict 
from federated_methods.fedmosaic_utils import (fedmosaic_homo_load_state_dict, fedmosaic_load_state_dict)


from federated_methods.feddat import feddat_create_trainer, feddat_hetero_load_state_dict, feddat_aggregate_state_dict
from federated_methods.feddistill import Distillation_aggregate_state_dict
from federated_methods.perada import perada_create_trainer
from federated_methods.fedsim import fedsim_create_trainer
from federated_methods.takfl import TAKFL_aggregate_state_dict
from federated_methods.fedmkt import FEDMKT_aggregate_state_dict
from federated_methods.feddpa import feddpa_create_trainer

def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    extra_modules = {}
    if 'sft' in mode:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['fedavg']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['ditto']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, perada_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['fedsim']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, fedsim_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['feddistill']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, Distillation_aggregate_state_dict
    elif mode in ['takfl']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, TAKFL_aggregate_state_dict
    elif mode in ['feddpa']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, feddpa_create_trainer, fedavg_aggregate_state_dict
    elif mode in ['feddat']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, feddat_hetero_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
    elif mode in ['perada']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedper_load_state_dict, perada_create_trainer, Distillation_aggregate_state_dict
    elif mode in ['fedmkt']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, FEDMKT_aggregate_state_dict
    elif mode in ['fedmosaic_homo']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedmosaic_homo_load_state_dict, fedmosaic_create_trainer, fedmosaic_aggregate_state_dict
    elif mode in ['fedmosaic']:
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedmosaic_load_state_dict, fedmosaic_create_trainer, fedmosaic_aggregate_state_dict
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
