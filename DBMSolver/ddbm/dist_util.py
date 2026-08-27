"""
Helpers for distributed training.
"""

import os

import torch
import torch.distributed as dist

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    torch.cuda.set_device(LOCAL_RANK)
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
