from typing import Callable, Tuple, Type, Dict

from selection_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer
from selection_methods.sft import sft_load_state_dict

from selection_methods.infoCluster import infoCluster_load_state_dict, infoCluster_aggregate_state_dict, infoCluster_create_trainer

from selection_methods.infoBatch import infobatch_create_trainer
from selection_methods.scoreBased import score_create_trainer

from selection_methods.divbs import divbs_create_trainer
from selection_methods.gradnorm import gradnorm_create_trainer

from selection_methods.sft_org import sft_org_create_trainer
from selection_methods.oasis import infogradsimv2_create_trainer



def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    extra_modules = {}
    if mode == 'sft':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'cluster':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, infoCluster_load_state_dict, infoCluster_create_trainer, infoCluster_aggregate_state_dict
    elif mode == 'score':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, score_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'infoBatch':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, infobatch_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'divbs':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, divbs_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'gradnorm':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, gradnorm_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'orin':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, infobudget_prob_create_trainer, fedavg_aggregate_state_dict
    elif mode == "sft_org":
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, sft_org_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'oasis':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, infogradsimv2_create_trainer,  fedavg_aggregate_state_dict
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules
