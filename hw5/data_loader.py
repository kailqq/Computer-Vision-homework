import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataLoader:
    """
    MNISTDataLoader class for loading the MNIST dataset
    """
    def __init__(self, batch_size=64, download=True):
        self.batch_size = batch_size
        self.download = download

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))# MNIST 数据集的均值和标准差
        ])

        self.train_dataset=None
        self.test_dataset=None
        self.train_loader=None
        self.test_loader=None

    def load_data(self):

        self.train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=self.download, 
            transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=self.download, 
            transform=self.transform
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.test_loader
    
    def show_data(self):
        # 显示训练集中的前10张图像
        plt.figure(figsize=(10, 3))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.axis('off')
            plt.imshow(self.train_dataset[i][0].numpy().squeeze(), cmap='gray')
            plt.title(f'Label: {self.train_dataset[i][1]}')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
