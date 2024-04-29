import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, precision_recall_curve, auc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import wandb
import pickle

wandb.init(project="c6_pretrained_resnet")

MIT_dataset = "/ghome/mcv/datasets/C3/MIT_split"
train_dataset_dir = "/ghome/group04/MCV-C5-G4/augmented_dataset"
test_dataset_dir = "/ghome/mcv/datasets/C3/MIT_small_train_1/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_HEIGHT = 224
IMG_WIDTH = 224

rotation_range = 30
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True

transform_train = transforms.Compose([
    transforms.RandomRotation(rotation_range),
    transforms.RandomHorizontalFlip(horizontal_flip),
    transforms.RandomVerticalFlip(horizontal_flip),
    transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=shear_range, shear=shear_range),
    transforms.RandomAffine(degrees=0, translate=(width_shift_range, height_shift_range)),
    transforms.RandomAffine(degrees=0, scale=(1 - zoom_range, 1 + zoom_range)),
    transforms.ToTensor(),
])

transform_retrieval = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root=train_dataset_dir, transform=transform_train)
test_dataset = ImageFolder(root=test_dataset_dir, transform=transform_retrieval)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(pretrained=True)

num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()

param_grid = {
    'lr': [0.001, 0.0005],
    'weight_decay': [0.01, 0.001],
    'freeze_layers': [True, False]
}

best_model = None
best_accuracy = 0.0

for params in ParameterGrid(param_grid):
    print(f"Training with parameters: {params}")
    
    if params['freeze_layers']:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    patience = 5
    early_stopping_counter = 0
    best_model_state_dict = copy.deepcopy(model.state_dict())
    
    precision_values = []
    recall_values = []

    for epoch in tqdm(range(20), desc="Training Progress", unit="epoch"):
        model.train()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in tqdm(train_dataloader, desc="Epoch Progress", unit="batch", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        average_loss = total_loss / len(train_dataloader)

        model.eval()
        test_loss = 0.0
        all_test_labels = []
        all_test_preds = []

        with torch.no_grad():
            for test_inputs, test_labels in tqdm(test_dataloader, desc="Test Progress", unit="batch", leave=False):
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)
                test_loss += criterion(test_outputs, test_labels).item()

                _, test_preds = torch.max(test_outputs, 1)
                all_test_labels.extend(test_labels.cpu().numpy())
                all_test_preds.extend(test_preds.cpu().numpy())

        test_accuracy = accuracy_score(all_test_labels, all_test_preds)
        average_test_loss = test_loss / len(test_dataloader)
        
        scheduler.step(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state_dict = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping!")
                break
    
        model.eval()
        all_scores = []
        all_labels = []
        binary_labels = []

        with torch.no_grad():
            for test_inputs, test_labels in tqdm(test_dataloader, desc="MAP Progress", unit="batch", leave=False):
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)
                probabilities = nn.functional.softmax(test_outputs, dim=1)
                scores = probabilities[:, 1].cpu().numpy()  
                all_scores.extend(scores)
                all_labels.extend(test_labels.cpu().numpy())

                binary_labels.extend([(1 if label == 1 else 0) for label in test_labels.cpu().numpy()])

        binary_labels = np.array([(1 if label == 1 else 0) for label in all_labels])

        map_score = average_precision_score(binary_labels, all_scores)

        binary_preds = [1 if score >= 0.5 else 0 for score in all_scores]

        prec_at_1 = precision_score(binary_labels, binary_preds, pos_label=1)
        prec_at_5 = precision_score(binary_labels, [1 if score >= sorted(all_scores, reverse=True)[4] else 0 for score in all_scores], pos_label=1)

        precision, recall, _ = precision_recall_curve(binary_labels, all_scores)
        
        precision_values.append(precision)
        recall_values.append(recall)

        wandb.log({"precision": precision, "recall": recall})

        wandb.log({
            "epoch": epoch + 1,
            "total_param": sum(p.numel() for p in model.parameters()),
            "train_loss": average_loss,
            "train_accuracy": accuracy,
            "test_loss": average_test_loss,
            "test_accuracy": test_accuracy,
            "MAP": map_score,
            "precision_at_1": prec_at_1,
            "precision_at_5": prec_at_5
        })

        print(f'Epoch {epoch + 1}/20 - Loss: {average_loss:.4f} - Accuracy: {accuracy:.4f} - Test Loss: {average_test_loss:.4f} - Test Accuracy: {test_accuracy:.4f} - MAP: {map_score:.4f} - Precision@1: {prec_at_1:.4f} - Precision@5: {prec_at_5:.4f}')

    plt.figure()
    for i in range(len(precision_values)):
        plt.plot(recall_values[i], precision_values[i], marker='.', label=f'Epoch {i+1}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(f'precision_recall_curve_{params}.png')
    plt.close()

    model.load_state_dict(best_model_state_dict)

    print(f'Best Accuracy: {best_accuracy:.4f} with parameters: {params}')

torch.save(model.state_dict(), "best_model.pth")

validation_embeddings = []
validation_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_dataloader, desc="Saving Validation Features", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        validation_embeddings.extend(outputs.cpu().numpy())
        validation_labels.extend(labels.cpu().numpy())

with open('validation_features.pkl', 'wb') as f:
    pickle.dump((validation_embeddings, validation_labels), f)
