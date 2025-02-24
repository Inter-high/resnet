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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    seed_everything(cfg['seed'])

    train_dataset, valid_dataset, test_dataset = get_datasets(cfg['seed'], cfg['data']['data_dir'])
    logger.info(f"Train Dataset: {len(train_dataset)} | Valid Dataset: {len(valid_dataset)} | Test Dataset: {len(test_dataset)}")
    train_loader, valid_loader, test_loader = get_loaders(train_dataset, valid_dataset, test_dataset, cfg['data']['batch_size'], cfg['data']['num_workers'])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = get_plain_network(n=cfg['model']['n'], num_classes=cfg['model']['num_classes'])
    model = get_resnet(n=cfg['model']['n'], num_classes=cfg['model']['num_classes'], shortcut_type=cfg['model']['type'])
    model = model.to(device)
    logger.info(f"Model Parameters: {count_model_parameters(model):,}")

    optimizer = optim.SGD(model.parameters(), lr=cfg['train']['lr'], momentum=cfg['train']['momentum'], weight_decay=cfg['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    trainer = Trainer(model, optimizer, scheduler, criterion, device, logger, hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    weight_path = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        "best_model.pth"
    )
    train_losses, train_accs, valid_losses, valid_accs, top1_error, top5_error = trainer.training(cfg['train']['epochs'], train_loader, valid_loader, test_loader, weight_path)

    pickle_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "training_results.pkl")
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
