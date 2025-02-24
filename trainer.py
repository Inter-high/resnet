import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, logger, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.lowest_loss = float("inf")
        self.device = device
        self.logger = logger
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
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

        return total_loss / len(train_loader), total_acc / total_samples

    def valid(self, valid_loader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
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

        return total_loss / len(valid_loader), total_acc / total_samples

    def test(self, test_loader):
        self.model.eval()
        top1_correct = 0.0
        top5_correct = 0.0
        total_samples = 0
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
        
        top1_error = 1 - (top1_correct / total_samples)
        top5_error = 1 - (top5_correct / total_samples)

        return top1_error, top5_error

    def training(self, epochs, train_loader, valid_loader, test_loader, weight_path):
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []
        
        for epoch in range(1, epochs + 1):
            # 185 에폭에 학습 종료
            if epoch > 185:
                break
            
            train_loss, train_acc = self.train(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 매 에폭 train metric은 항상 기록
            loss_dict = {"train": train_loss}
            acc_dict = {"train": train_acc}
            
            if epoch % 5 == 0:
                valid_loss, valid_acc = self.valid(valid_loader)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)
                
                # valid metric도 함께 기록
                loss_dict["valid"] = valid_loss
                acc_dict["valid"] = valid_acc
                
                self.logger.info(f"Epoch: {epoch}/185 | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
                
                # valid_loss가 self.lowest_loss보다 낮으면 갱신하고 가중치 저장
                if valid_loss < self.lowest_loss:
                    self.lowest_loss = valid_loss
                    torch.save(self.model.state_dict(), weight_path)
                    self.logger.info(f"New lowest valid loss: {valid_loss:.4f}. Model weights saved to {weight_path}")
            
            # 두 개의 그래프("Loss", "Acc")에 각각 train과 valid 값을 함께 기록
            self.writer.add_scalars("Loss", loss_dict, epoch)
            self.writer.add_scalars("Acc", acc_dict, epoch)
            
            # 에폭마다 scheduler step 호출 (즉, 에폭 단위 스케줄러 업데이트)
            self.scheduler.step()
        
        top1_error, top5_error = self.test(test_loader)
        self.logger.info(f"Top1 Error: {top1_error:.4f} | Top5 Error: {top5_error:.4f}")
        
        return train_losses, train_accs, valid_losses, valid_accs, top1_error, top5_error
        