import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_csv(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print("File does not exist")
        sys.exit(1)

    # Read the csv file
    df = pd.read_csv(file_path)
    df_train_loss = df.dropna(subset=['train_loss_epoch'])
    df_val_loss = df.dropna(subset=['val_loss_epoch'])

    # Plot the data
    plt.plot(df_train_loss['epoch'], df_train_loss['train_loss_epoch'], label='Training Loss')
    plt.plot(df_val_loss['epoch'], df_val_loss['val_loss_epoch'], label='Validation Loss')
    # Add legend
    plt.legend()
    # label axes and add title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig("training loss.png")

file_path = '/home/georg/projects/university/C5/task3/task_3e/logs/TripletNetworkCSV/version_19/metrics.csv'
plot_csv(file_path)