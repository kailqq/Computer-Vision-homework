import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt

from model import LeNet
from data_loader import MNISTDataLoader
from config import Config


class Trainer:
    """
    Trainer class for training the model
    """
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=self.config.step_size, 
                                                   gamma=self.config.gamma)

        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.plot_dir, exist_ok=True)

        #remove all files in the log_dir
        for file in os.listdir(self.config.log_dir):
            os.remove(os.path.join(self.config.log_dir, file))
    
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        self.best_acc = 0.0
        self.best_model_path = os.path.join(self.config.save_dir, "best_model.pth")

        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []

    def train_per_epoch(self,epoch):
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        train_total = 0

        for i, (images, labels) in enumerate(self.train_loader, 0):
            images = images.to(self.device)
            labels = labels.to(self.device)

            #forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            #backward
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (i + 1) % self.config.log_interval == 0:
                print(f"epoch {epoch+1}, Step {i + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        train_loss /= len(self.train_loader)
        train_acc = train_correct / train_total
        return train_loss, train_acc

    def test(self):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader, 0):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_loss /= len(self.test_loader)
        test_acc = test_correct / test_total

        return test_loss, test_acc
    
    def train(self):

        print(f"Training on {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Step size: {self.config.step_size}")
        print(f"Gamma: {self.config.gamma}")
        print("-"*100)

        best_acc = 0.0
        start_time = time.time()

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_per_epoch(epoch)
            test_loss, test_acc = self.test()
            
            self.writer.add_scalar("train/train_loss", train_loss, epoch)
            self.writer.add_scalar("train/train_acc", train_acc, epoch)
            self.writer.add_scalar("test/test_loss", test_loss, epoch)
            self.writer.add_scalar("test/test_acc", test_acc, epoch)

            self.scheduler.step()
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), self.best_model_path)
            
            end_time = time.time()
            
            if (epoch + 1) % self.config.epochs_log_interval == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs},Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Time: {end_time - start_time:.2f}s")
        
        print("-"*50)

    def save_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.config.plot_dir, "loss.png"))
        plt.close()

def main():
    config = Config()
    loader = MNISTDataLoader()
    train_loader, test_loader = loader.load_data()
    loader.show_data()
    model = LeNet()
    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()
    trainer.save_results()

if __name__ == "__main__":
    main()


