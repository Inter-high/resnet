"""
Utility functions for loading and preprocessing the CIFAR-10 dataset using PyTorch and torchvision.

Author: yumemonzo@gmail.com
Date: 2025-02-24
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns the training and testing transforms for the CIFAR-10 dataset.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: A tuple containing the training transform and testing transform.
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    return train_transform, test_transform


def create_train_valid_split(seed: int, train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset) -> Tuple[Subset, Subset]:
    """
    Splits the training dataset into training and validation subsets.

    Args:
        seed (int): Random seed for reproducibility.
        train_dataset (Dataset): The training dataset with training transforms.
        valid_dataset (Dataset): The training dataset with testing transforms.

    Returns:
        Tuple[Subset, Subset]: The training and validation subsets.
    """
    num_samples = len(train_dataset)
    train_size = int(0.9 * num_samples)
    valid_size = num_samples - train_size

    dummy_dataset = list(range(num_samples))
    generator = torch.Generator().manual_seed(seed)
    train_idx_dataset, valid_idx_dataset = random_split(dummy_dataset, [train_size, valid_size], generator=generator)

    train_indices = list(train_idx_dataset)
    valid_indices = list(valid_idx_dataset)

    train_dataset = Subset(train_dataset, train_indices)
    valid_dataset = Subset(valid_dataset, valid_indices)
    
    return train_dataset, valid_dataset


def get_datasets(seed: int, data_dir: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads and prepares the CIFAR-10 datasets for training, validation, and testing.

    Args:
        seed (int): Random seed for dataset splitting.
        data_dir (str): Directory to store the CIFAR-10 data.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validation, and testing datasets.
    """
    train_transform, test_transform = get_transforms()

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    valid_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
    train_dataset, valid_dataset = create_train_valid_split(seed, train_dataset, valid_dataset)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates data loaders for the training, validation, and testing datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        valid_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Testing dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation, and testing data loaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader
