import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transformsv2
import torchvision.models as models
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt

from siamese_dataloader import get_dataloader
import warnings

from sklearn.metrics import average_precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
            target.float() * distances
            + (1 + -1 * target).float()
            * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        return losses.mean() if size_average else losses.sum()


transform = transformsv2.Compose(
    [
        transformsv2.RandomResizedCrop(size=(224, 224), antialias=True),
        transformsv2.ToTensor(),
        transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dir = "/ghome/group04/MCV-C5-G4/MIT_small_train_1/test"
val_dir = "/ghome/group04/MCV-C5-G4/MIT_small_train_1/test"
test_dir = "/ghome/group04/MCV-C5-G4/MIT_small_train_1/test"
model_dir = "/ghome/group04/MCV-C5-G4/week3/siamese"

device = "cuda" if torch.cuda.is_available() else "cpu"
train_batch_size = 32
test_val_batch_size = 64
num_epoch = 10
num_workers = 3

print(f"Device: {device}")

train_dataloader, test_dataloader, val_dataloader = get_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    val_dir=val_dir,
    train_batch_size=train_batch_size,
    test_val_batch_size=test_val_batch_size,
    num_workers=num_workers,
    transform=transform,
)


def train_validation_test(train_dataloader, val_dataloader, test_dataloader):

    # load model
    model = models.resnet18(pretrained=True)
    model.to(device=device)
    print("resnet18 model loaded")

    # TRAINING LOOP
    contrastive_loss = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epoch)):
        running_loss = 0.0
        running_accuracy = 0.0
        model.train()
        for data in tqdm(train_dataloader, total=len(train_dataloader)):
            img1, img2, target = data
            img1, img2, target = (
                img1.to(device),
                img2.to(device),
                target.to(device),
            )

            optimizer.zero_grad()

            img1_feature_vector = model(img1)
            img2_feature_vector = model(img2)

            loss = contrastive_loss(img1_feature_vector, img2_feature_vector, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for data in tqdm(val_dataloader, total=len(val_dataloader)):
                img1, img2, target = data
                img1, img2, target = (
                    img1.to(device),
                    img2.to(device),
                    target.to(device),
                )

                img1_feature_vector = model(img1)
                img2_feature_vector = model(img2)

                loss = contrastive_loss(
                    img1_feature_vector, img2_feature_vector, target
                )
                val_loss += loss.item()

                # Calculate Euclidean distances
                distance = torch.norm(img1_feature_vector - img2_feature_vector, dim=1)

                # Increment correct predictions if the positive is closer to the anchor than the negative
                correct_predictions += torch.sum(distance < 0.5).item()
                total_predictions += img1.size(0)

            avg_val_loss = val_loss / len(val_dataloader)
            validation_accuracy = correct_predictions / total_predictions
            print(
                f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}"
            )

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epoch}, Average Loss: {avg_loss:.4f}")

    # TESTING
    model.eval()
    test_loss = 0.0
    total_examples = 0

    # For Recall and F1 Score
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    with torch.no_grad():

        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            img1, img2, target = data
            img1, img2, target = (
                img1.to(device),
                img2.to(device),
                target.to(device),
            )

            img1_feature_vector = model(img1)
            img2_feature_vector = model(img2)

            loss = contrastive_loss(img1_feature_vector, img2_feature_vector, target)
            test_loss += loss.item()

            # Compute distances
            distance = torch.norm(img1_feature_vector - img2_feature_vector, dim=1)

            # Increment correct predictions if the positive is closer to the anchor than the negative
            correct_predictions = distance < 0.5  # Adjust threshold as needed
            true_positives += correct_predictions.sum().item()
            false_negatives += (~correct_predictions).sum().item()
            false_positives += (
                correct_predictions.sum().item()
            )  # Simplified, usually you'd have more negatives to consider

            total_predictions += img1.size(0)

    # Recall and F1 Score
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("-" * 50)

    print(f"Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}")

    # Print test loss and accuracy
    print(f"Test Loss: {test_loss / total_predictions}")
    test_accuracy = true_positives / total_predictions
    print(f"Test Accuracy: {test_accuracy}")

    # Save the model file
    model_pt_path = os.path.join(model_dir, "siamese_resnet18.pt")
    torch.save(model, model_pt_path)
    print(f"Model saved at {model_pt_path}")


if __name__ == "__main__":
    train_validation_test(
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
