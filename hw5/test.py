import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from model import LeNet
from data_loader import MNISTDataLoader
from config import Config

class Tester:
    """
    Tester class for testing the model
    """
    def __init__(self,model_path, test_loader, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self.load_model()
        self.test_loader = test_loader
        self.config = config
        self.results_dir = f"{self.config.plot_dir}/test_results"
        os.makedirs(self.results_dir, exist_ok=True)
       

    def load_model(self):
        model = LeNet()
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        return model
    
    def test(self):
        self.model.eval()
        test_correct = 0
        test_total = 0
        cnt = 0;
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader, 0):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

                # 随机保存十张测试结果
                if cnt < 10 and torch.rand(1).item() < 0.1:
                    plt.figure(figsize=(2, 2))
                    plt.imshow(images[0].cpu().numpy().squeeze(), cmap='gray')
                    plt.title(f'Pred: {predicted[0].item()}, True: {labels[0].item()}')
                    plt.axis('off')
                    plt.savefig(f'{self.results_dir}/test_result_{cnt}.png')
                    plt.close()
                    cnt += 1
        acc = 100 * test_correct / test_total
        print(f"Test Accuracy: {acc:.2f}%")
        return acc
    
    
def main():
    config = Config()
    model_path = f"{config.save_dir}/best_model.pth"
    _, test_loader = MNISTDataLoader().load_data()
    tester = Tester(model_path, test_loader, config)
    tester.test()

if __name__ == "__main__":
    main()

        
