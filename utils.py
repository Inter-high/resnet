"""
Utility functions for reproducibility, model parameter counting, and plotting performance comparisons.

Author: yumemonzo@gmail.com
Date: 2025-02-24
"""

import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any


def seed_everything(seed: int = 42) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        
    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_compare_loss(data: List[List[float]], labels: List[str], save_path: str, title: str = 'Loss Comparison') -> None:
    """
    Plot loss curves from multiple experiments and save the figure.
    
    Args:
        data (List[List[float]]): A list where each element is a list of loss values for an experiment.
        labels (List[str]): A list of labels corresponding to each experiment.
        save_path (str): File path to save the plot (e.g., 'loss_comparison.png').
        title (str): Title of the plot. Defaults to 'Loss Comparison'.
        
    Raises:
        ValueError: If the lengths of data and labels do not match.
    """
    if len(data) != len(labels):
        raise ValueError("The lengths of data and labels must be equal.")
        
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(data):
        epochs = range(1, len(d) + 1)
        plt.plot(epochs, d, label=labels[i], linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


def plot_compare_acc(data: List[List[float]], labels: List[str], save_path: str, title: str = 'Accuracy Comparison') -> None:
    """
    Plot accuracy curves from multiple experiments and save the figure.
    
    Args:
        data (List[List[float]]): A list where each element is a list of accuracy values for an experiment.
        labels (List[str]): A list of labels corresponding to each experiment.
        save_path (str): File path to save the plot (e.g., 'accuracy_comparison.png').
        title (str): Title of the plot. Defaults to 'Accuracy Comparison'.
        
    Raises:
        ValueError: If the lengths of data and labels do not match.
    """
    if len(data) != len(labels):
        raise ValueError("The lengths of data and labels must be equal.")
    
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(data):
        epochs = range(1, len(d) + 1)
        plt.plot(epochs, d, label=labels[i], linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


def plot_compare_error(errors: List[float], labels: List[str], save_path: str, title: str = 'Error Comparison') -> None:
    """
    Plot a bar chart comparing error values from multiple experiments and save the figure.
    
    Args:
        errors (List[float]): A list of error values, one for each experiment.
        labels (List[str]): A list of labels corresponding to each experiment.
        save_path (str): File path to save the plot (e.g., 'error_comparison.png').
        title (str): Title of the plot. Defaults to 'Error Comparison'.
        
    Raises:
        ValueError: If the lengths of errors and labels do not match.
    """
    if len(errors) != len(labels):
        raise ValueError("The lengths of errors and labels must be equal.")
    
    n = len(errors)
    x = np.arange(n)
    width = 0.5
    colors = [f'C{i}' for i in range(n)]
    
    plt.figure(figsize=(8, 6))
    plt.bar(x, errors, width, color=colors)
    plt.xticks(x, labels)
    plt.ylabel('Error')
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()
