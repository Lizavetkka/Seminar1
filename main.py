import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import nn
from torchvision.transforms import ToTensor

from scripts.dataset import get_data_loaders
from scripts.model import NeuralNetwork
from scripts.train import test_loop, train_loop

# Hyperparameters to experiment with
BATCH_SIZES = [32, 64]
LEARNING_RATES = [0.0001 ,0.001, 0.01]
OPTIMIZERS = [torch.optim.SGD, torch.optim.Adam]
MOMENTUMS = [0,0.9]
EPOCHS = 5
loss_fn = nn.CrossEntropyLoss()
transform = ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Using {device} device")

# Create directories if they don't exist
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('results/images', exist_ok=True)
os.makedirs('results/models', exist_ok=True)

# Function to plot training and validation metrics
def plot_metrics(train_loss, val_loss, train_accuracy, val_accuracy, lr, optimizer_name, momentum, batch_size):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss - LR: {lr}, Optimizer: {optimizer_name}, Momentum: {momentum}, Batch Size: {batch_size}", fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Train Accuracy', color='blue')
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy - LR: {lr}, Optimizer: {optimizer_name}, Momentum: {momentum}, Batch Size: {batch_size}", fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/images/accuracy_lr_{lr}_opt_{optimizer_name}_momentum_{momentum}_batchsize_{batch_size}.png')

    plt.close()

if __name__ == "__main__":

    best_models = []  
    best_performances = []  

    for batch_size in BATCH_SIZES:
        for lr in LEARNING_RATES:
            for optimizer_class in OPTIMIZERS:
                for momentum in MOMENTUMS if optimizer_class == torch.optim.SGD else [None]:
                    
                    model = NeuralNetwork().to(device)
                    train_dataloader, test_dataloader = get_data_loaders(batch_size, transform)

                    if optimizer_class == torch.optim.SGD:
                        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=momentum)
                    else:
                        optimizer = optimizer_class(model.parameters(), lr=lr)

                    train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []
                    total_time = 0

                    for epoch in range(EPOCHS):
                        start_time = time.time()  

                        train_logs = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
                        train_loss.append(train_logs['loss'])
                        train_accuracy.append(train_logs['accuracy'])

                        test_logs = test_loop(test_dataloader, model, loss_fn)
                        val_loss.append(test_logs['loss'])
                        val_accuracy.append(test_logs['accuracy'])

                        epoch_time = time.time() - start_time
                        total_time += epoch_time

                    avg_val_accuracy = sum(val_accuracy) / len(val_accuracy)

                    # Update best models and performances
                    if len(best_performances) < 3 or avg_val_accuracy > min(best_performances):
                        if len(best_performances) >= 3:
                            worst_index = best_performances.index(min(best_performances))
                            del best_performances[worst_index]
                            del best_models[worst_index]

                        best_models.append({
                            'state_dict': model.state_dict(),
                            'lr': lr,
                            'optimizer': optimizer_class.__name__,
                            'momentum': momentum,
                            'batch_size': batch_size
                        })
                        best_performances.append(avg_val_accuracy)

                    # Save metrics to a file
                    with open(f'results/metrics/metrics_lr_{lr}_optimizer_{optimizer_class.__name__}_momentum_{momentum}_batchsize_{batch_size}.txt', 'w') as f:
                        f.write(f"Learning Rate: {lr}, Optimizer: {optimizer_class.__name__}, Momentum: {momentum}, Batch Size: {batch_size}\n")
                        f.write(f"Final Train Loss: {train_logs['loss']}, Train Accuracy: {train_logs['accuracy']}\n")
                        f.write(f"Final Val Loss: {test_logs['loss']}, Val Accuracy: {test_logs['accuracy']}\n")
                        f.write(f"Total Training Time: {total_time:.2f}s\n")

                    # Save model
                    torch.save(model.state_dict(), f'results/models/model_lr_{lr}_opt_{optimizer_class.__name__}_momentum_{momentum}_batchsize_{batch_size}.pth')

                    # Plot metrics
                    plot_metrics(train_loss, val_loss, train_accuracy, val_accuracy, lr, optimizer_class.__name__, momentum, batch_size)

    # Display best models
    for i, (performance, model_info) in enumerate(zip(best_performances, best_models)):
        print(f"Best Model {i + 1}: Val Accuracy = {performance:.4f}")
        print(f"Hyperparameters: LR={model_info['lr']}, Optimizer={model_info['optimizer']}, Momentum={model_info['momentum']}, Batch Size={model_info['batch_size']}")

