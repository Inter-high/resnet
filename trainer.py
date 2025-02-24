"""
This file defines the Trainer class for training, validating, and testing a model using PyTorch.
It logs metrics via TensorBoard and saves the best model based on validation loss.

Author: yumemonzo@gmail.com
Date: 2025-02-24
"""

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List


class Trainer:
    """
    Trainer class to manage training, validation, and testing processes.
    
    Attributes:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        criterion (torch.nn.Module): Loss function.
        device (str): Device to perform computations ('cuda' or 'cpu').
        logger: Logger for recording training progress.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        lowest_loss (float): Tracks the lowest validation loss for saving the best model.
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, 
                 criterion: torch.nn.Module, device: str, logger, log_dir: str) -> None:
        """
        Initializes the Trainer.
        
        Args:
            model (torch.nn.Module): Model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler: Learning rate scheduler.
            criterion (torch.nn.Module): Loss function.
            device (str): Device to use ('cuda' or 'cpu').
            logger: Logger for logging information.
            log_dir (str): Directory for TensorBoard logs.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.lowest_loss: float = float("inf")
        self.device: str = device
        self.logger = logger
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Trains the model for one epoch.
        
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            
        Returns:
            Tuple[float, float]: Average training loss and accuracy.
        """
        self.model.train()
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_samples: int = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=True)

        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_samples += y.size(0)
            pred = y_hat.argmax(dim=1)
            total_acc += (pred == y).sum().item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

    def valid(self, valid_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validates the model.
        
        Args:
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            
        Returns:
            Tuple[float, float]: Average validation loss and accuracy.
        """
        self.model.eval()
        total_loss: float = 0.0
        total_acc: float = 0.0
        total_samples: int = 0
        progress_bar = tqdm(valid_loader, desc="Validating", leave=True)

        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                total_loss += loss.item()
                total_samples += y.size(0)
                pred = y_hat.argmax(dim=1)
                total_acc += (pred == y).sum().item()

                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(valid_loader)
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

    def test(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Tests the model and computes the top-1 and top-5 error rates.
        
        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.
            
        Returns:
            Tuple[float, float]: Top-1 error and top-5 error.
        """
        self.model.eval()
        top1_correct: float = 0.0
        top5_correct: float = 0.0
        total_samples: int = 0
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)

        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)

                _, top5_preds = torch.topk(y_hat, k=5, dim=1)
                top1_pred = top5_preds[:, 0]
                total_samples += y.size(0)

                top1_correct += (top1_pred == y).sum().item()
                top5_correct += sum(y[i].item() in top5_preds[i].tolist() for i in range(y.size(0)))

                progress_bar.set_postfix(top1_acc=top1_correct / total_samples, top5_acc=top5_correct / total_samples)

        top1_error: float = 1 - (top1_correct / total_samples)
        top5_error: float = 1 - (top5_correct / total_samples)
        return top1_error, top5_error

    def training(self, epochs: int, train_loader: torch.utils.data.DataLoader, 
                 valid_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, 
                 weight_path: str) -> Tuple[List[float], List[float], List[float], List[float], float, float]:
        """
        Runs the training loop, validates periodically, and tests after training.
        Stops training after 185 epochs.
        
        Args:
            epochs (int): Total number of epochs to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for test data.
            weight_path (str): Path to save the best model weights.
            
        Returns:
            Tuple containing:
                - List of training losses per epoch.
                - List of training accuracies per epoch.
                - List of validation losses (recorded every 5 epochs).
                - List of validation accuracies (recorded every 5 epochs).
                - Top-1 error on the test set.
                - Top-5 error on the test set.
        """
        train_losses: List[float] = []
        train_accs: List[float] = []
        valid_losses: List[float] = []
        valid_accs: List[float] = []

        for epoch in range(1, epochs + 1):
            if epoch > 185:
                break

            train_loss, train_acc = self.train(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            loss_dict = {"train": train_loss}
            acc_dict = {"train": train_acc}

            if epoch % 5 == 0:
                valid_loss, valid_acc = self.valid(valid_loader)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)
                loss_dict["valid"] = valid_loss
                acc_dict["valid"] = valid_acc

                self.logger.info(f"Epoch: {epoch}/185 | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

                if valid_loss < self.lowest_loss:
                    self.lowest_loss = valid_loss
                    torch.save(self.model.state_dict(), weight_path)
                    self.logger.info(f"New lowest valid loss: {valid_loss:.4f}. Model weights saved to {weight_path}")

            self.writer.add_scalars("Loss", loss_dict, epoch)
            self.writer.add_scalars("Acc", acc_dict, epoch)
            self.scheduler.step()

        top1_error, top5_error = self.test(test_loader)
        self.logger.info(f"Top1 Error: {top1_error:.4f} | Top5 Error: {top5_error:.4f}")

        return train_losses, train_accs, valid_losses, valid_accs, top1_error, top5_error
