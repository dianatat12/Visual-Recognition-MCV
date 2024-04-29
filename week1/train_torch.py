import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchmetrics.functional import accuracy
import wandb
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy

# data dir
train_dir = "../augmented_dataset"
test_dir = "../MIT_small_train_1/train"
validation_dir = "../test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"selected device : {device}\n\n")
train_batch_size = 64
test_validation_batch_size = 128
num_epochs = 100

ES_MONITOR = "val_accuracy"
ES_MODE = "max"
ES_MIN_DELTA = 0.01
ES_PATIENCE = 3
ES_RESTORE_BEST = True

LR_MONITOR = "val_accuracy"
LR_MODE = "max"
LR_MIN_DELTA = 0.01
LR_PATIENCE = 3
LR_FACTOR = 0.25
LR_MIN_LR = 0.00000001


# DEFINE MODEL
class Model_11(nn.Module):
    def __init__(self, IMG_CHANNEL, NUM_CLASSES):
        super(Model_11, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


# Dataloader functions
def get_dataloader(root_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(root=root_dir, transform=transform)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )
    return dataloader


class EarlyStopping:
    def __init__(
        self,
        patience=ES_PATIENCE,
        verbose=True,
        delta=ES_MIN_DELTA,
        path="checkpoint.pt",
        trace_func=print,
        restore_best_weights=ES_RESTORE_BEST,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.restore_best_weights = restore_best_weights
        if restore_best_weights:
            self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        if self.restore_best_weights:
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


if __name__ == "__main__":
    model = Model_11(IMG_CHANNEL=3, NUM_CLASSES=8)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=LR_MODE,
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=True,
        min_lr=LR_MIN_LR,
        threshold=LR_MIN_DELTA,
        cooldown=0,
    )
    early_stopper = EarlyStopping(
        patience=ES_PATIENCE,
        verbose=True,
        delta=ES_MIN_DELTA,
        path="model_checkpoint.pt",
    )

    train_dataloader = get_dataloader(train_dir, train_batch_size)
    test_dataloader = get_dataloader(test_dir, test_validation_batch_size)
    val_dataloader = get_dataloader(validation_dir, test_validation_batch_size)

    wandb.init(project="MCV-C5-WEEK-1")

    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(
            train_dataloader, total=len(train_dataloader), leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}"
        )
        wandb.log({"Epoch":epoch+1,"Train Loss": epoch_loss, "Train Accuracy": train_accuracy,"Framework":"PyTorch"})

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct_predictions / total_predictions
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}")
        wandb.log({"Epoch":epoch+1,"Validation Loss": val_loss, "Validation Accuracy": val_accuracy,"Framework":"PyTorch"})

        # Step the scheduler on validation loss
        scheduler.step(val_loss)

        # Early Stopping check
        # early_stopper(val_loss, model)
        # if early_stopper.early_stop:
        #     print("Early stopping triggered")
        #     break

    # Test phase (unchanged from your original code)
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)  # Assuming a classification problem
            acc = (preds == labels).float().mean()

            test_acc += acc.item() * inputs.size(0)

    test_loss /= len(test_dataloader.dataset)
    test_acc /= len(test_dataloader.dataset)
    wandb.log({"Test Loss": test_loss, "Test Accuracy": test_acc,"Framework":"PyTorch"})

    print(f"TEST LOSS: {test_loss:.4f} | TEST ACCURACY: {test_acc:.4f}")
