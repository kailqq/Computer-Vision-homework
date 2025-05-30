import random

class Config:
    """
    Config class for the model
    """
    def __init__(self):
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 0.001
        self.step_size = 7
        self.gamma = 0.1
        self.save_dir = "./checkpoints"
        self.log_dir = "./logs"
        self.plot_dir = "./plots"
        self.log_interval =300
        self.epochs_log_interval = 1
        self.seed = random.randint(0, 10000)
