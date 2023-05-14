"""
Generic utility functions
"""

import numpy as np
import torch
import random
from pathlib import Path
from itertools import repeat


def prepare_device(n_gpu_use):
    """
    Set up GPU device if available, and transfer model into configured device.

    :param n_gpu_use: int, number of GPUs used
    :return: tuple, device to transfer to, list of GPU ids
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use"
              " is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def ensure_dir(dirname):
    """
    Check whether given directory was created; if not, the function will create the directory.

    :param dirname: str, path to the directory
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def inf_loop(data_loader):
    """
    Wrapper function to reiterate infinitely through given data loader.

    :param data_loader: DataLoader object, iterable over the dataset
    """
    for loader in repeat(data_loader):
        yield from loader

def set_seed(seed=1):
    """
    Set seed for reproducibility of results (applied to numpy and pytorch).

    :param seed: int, seed number
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

