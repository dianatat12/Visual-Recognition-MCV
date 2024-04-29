import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

class PlotMetricsCallback(Callback):
    def __init__(self, csv_path, save_dir):
        super().__init__()
        self.csv_path = csv_path
        self.save_dir = save_dir

    def on_train_end(self, trainer, pl_module):
        # Load CSV logs
        df = pd.read_csv(self.csv_path)
        df = df.interpolate(method='linear', axis=0)
        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_loss_epoch'], label='Training Loss')
        plt.plot(df['epoch'], df['val_loss_epoch'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/loss_plot.png')
        plt.close()

        # Plot training and validation accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_accuracy_epoch'], label='Training Accuracy')
        plt.plot(df['epoch'], df['val_accuracy_epoch'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/accuracy_plot.png')
        plt.close()