"""Utils for evaluating robot policies in the LIBERO simulator."""

import os
import random
import time

import numpy as np
import torch

from experiments.robot.openvla_utils import get_vla, get_vla_action

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg):
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """Gets image resize size for a model class (square)."""
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_decoding_params(cfg):
    """Collect the decoding hyperparameters relevant to the selected `decoding_mode`."""
    mode = cfg.decoding_mode
    if mode == "greedy":
        return {}
    if mode == "temp":
        return {"temperature": cfg.temperature}
    if mode == "topk":
        return {"temperature": cfg.temperature, "top_k": cfg.top_k}
    if mode == "topp":
        return {"top_p": cfg.top_p}
    if mode == "scale":
        return {
            "T0": cfg.T0,
            "kappa": cfg.kappa,
            "alpha": cfg.alpha,
            "epsilon": cfg.epsilon,
            "num_logits": cfg.num_logits,
            "attn_sensitivity": cfg.attn_sensitivity,
        }
    raise ValueError(f"Unknown decoding_mode: {mode}")


def get_action(cfg, model, obs, task_label, processor=None, uncertainty_history=None):
    """Queries the model to get a single action."""
    if cfg.model_family != "openvla":
        raise ValueError("Unexpected `model_family` found in config.")

    action, info = get_vla_action(
        model,
        processor,
        cfg.pretrained_checkpoint,
        obs,
        task_label,
        cfg.unnorm_key,
        center_crop=cfg.center_crop,
        decoding_mode=cfg.decoding_mode,
        decoding_params=get_decoding_params(cfg),
        uncertainty_history=uncertainty_history,
    )
    assert action.shape == (ACTION_DIM,)
    return action, info


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
