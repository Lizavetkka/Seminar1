import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from scripts.dataset import get_data_loaders
from scripts.train import test_loop, train_loop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_classes = 10  
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
epochs = 5 

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(batch_size, transform)

    best_accuracy = 0.0  
    model_save_path = "best_model.pth"  

    train_losses = []  
    train_accuracies = []  
    test_losses = []  
    test_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_metrics = train_loop(train_loader, model, loss_fn, optimizer, batch_size=32)
        test_metrics = test_loop(test_loader, model, loss_fn)

        print(f"Train Accuracy: {train_metrics['accuracy'] * 100:.2f}%")
        print(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")

        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with accuracy: {best_accuracy * 100:.2f}%")

        train_losses.append(train_metrics['loss'])
        train_accuracies.append(train_metrics['accuracy'])
        test_losses.append(test_metrics['loss'])
        test_accuracies.append(test_metrics['accuracy'])

        scheduler.step() 

    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in train_accuracies], label='Train Accuracy')
    plt.plot(epochs_range, [acc * 100 for acc in test_accuracies], label='Test Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png') 
 