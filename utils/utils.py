import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA, if applicable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def ensure_dir(directory):
    """
    Create directory if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_device():
    """
    Get the device (MPS for MacBook, CUDA for GPU, CPU otherwise).

    Returns:
        torch.device
    """
    if torch.backends.mps.is_available():
        print("Using MPS backend.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    else:
        print("Using CPU backend.")
        return torch.device("cpu")


def labels_to_names(labels, task_list):
    """
    Converts integer labels to task names.

    Args:
        labels (list or np.array): List of integer labels.
        task_list (list): List of task names.

    Returns:
        list of str: Corresponding task names.
    """
    return [task_list[label] for label in labels]


def names_to_labels(names, task_list):
    """
    Converts task names to integer labels.

    Args:
        names (list): List of task names.
        task_list (list): List of all possible task names.

    Returns:
        list of int: Corresponding integer labels.
    """
    return [task_list.index(name) for name in names]
