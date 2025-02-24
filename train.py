"""
Main training script for CIFAR-10 experiments using Hydra configuration.

This script sets up data loaders, constructs the model (either a plain network or ResNet),
defines the optimizer, loss function, and learning rate scheduler, and trains the model using the Trainer class.
It also saves the best model weights and training results.

Author: yumemonzo@gmail.com
Date: 2025-02-24
"""

import os
import logging
import pickle
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from data import get_datasets, get_loaders
from model import get_plain_network, get_resnet
from trainer import Trainer
from utils import seed_everything, count_model_parameters

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """
    Sets up and runs the training pipeline for CIFAR-10.

    Args:
        cfg (DictConfig): Hydra configuration object containing training, data, and model parameters.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed_everything(cfg['seed'])

    # Load datasets and create data loaders
    train_dataset, valid_dataset, test_dataset = get_datasets(cfg['seed'], cfg['data']['data_dir'])
    logger.info(f"Train Dataset: {len(train_dataset)} | Valid Dataset: {len(valid_dataset)} | Test Dataset: {len(test_dataset)}")
    train_loader, valid_loader, test_loader = get_loaders(
        train_dataset,
        valid_dataset,
        test_dataset,
        cfg['data']['batch_size'],
        cfg['data']['num_workers']
    )

    # Select device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model (uncomment plain network if needed)
    if cfg['model']['resnet']:
        model = get_resnet(n=cfg['model']['n'], num_classes=cfg['model']['num_classes'], shortcut_type=cfg['model']['type'])
    else:
        model = get_plain_network(n=cfg['model']['n'], num_classes=cfg['model']['num_classes'])
    model = model.to(device)
    logger.info(f"Model Parameters: {count_model_parameters(model):,}")

    # Define optimizer, loss criterion, and learning rate scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)

    # Initialize Trainer and start training
    output_dir: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    trainer = Trainer(model, optimizer, scheduler, criterion, device, logger, output_dir)

    weight_path: str = os.path.join(output_dir, "best_model.pth")
    train_losses, train_accs, valid_losses, valid_accs, top1_error, top5_error = trainer.training(
        cfg['train']['epochs'], train_loader, valid_loader, test_loader, weight_path
    )

    # Save training results to a pickle file
    pickle_path: str = os.path.join(output_dir, "training_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "train_losses": train_losses,
            "train_accs": train_accs,
            "valid_losses": valid_losses,
            "valid_accs": valid_accs,
            "top1_error": top1_error,
            "top5_error": top5_error
        }, f)

if __name__ == "__main__":
    my_app()
