import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transformsv2
import torchvision.models as models
from tqdm import tqdm


# OPTUNA & W&B
import optuna
from sklearn.metrics import average_precision_score

# local imoprts
from utils import get_dataloader, get_feature_extractor_model, calculate_distance
import warnings

# ignore warnings
warnings.filterwarnings("ignore")

# data dir
TRAIN_DIR = "/home/mimo/Desktop/MS CV/C5/data/MIT_small_train_2/train"
VALIDATION_DIR = "/home/mimo/Desktop/MS CV/C5/data/MIT_small_train_2/test"
TEST_DIR = "/home/mimo/Desktop/MS CV/C5/data/MIT_small_train_1/test"
MODEL_DIR = "./models"
if os.path.exists(MODEL_DIR) is False:
    os.makedirs(MODEL_DIR)

# param
device = "cuda" if torch.cuda.is_available() else "cpu"
train_batch_size = 16
test_val_batch_size = 64
num_epoch = 25
num_workers = 3

print(f"DEVICE BEING USED IS {device}")
# transforms
transforms = transformsv2.Compose(
    [
        transformsv2.RandomResizedCrop(size=(224, 224), antialias=True),
        transformsv2.ToTensor(),
        transformsv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# dataloader
train_dataloader, test_dataloader, val_dataloader = get_dataloader(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    val_dir=VALIDATION_DIR,
    train_batch_size=train_batch_size,
    test_val_batch_size=test_val_batch_size,
    num_workers=num_workers,
    transforms=transforms,
)


def train_validation_test(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    model_name,
):

    # load model
    model = get_feature_extractor_model(model_name)
    model.to(device=device)
    print("MODEL IS LOADED !!")

    # TRAINING LOOP
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epoch)):
        running_loss = 0.0
        model.train()
        for data in tqdm(
            train_dataloader, total=len(train_dataloader), desc="BATCH TRAINING "
        ):
            anchor, positive, negative = data
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            optimizer.zero_grad()

            anchor_feature_vector = model(anchor)
            positive_feature_vector = model(positive)
            negative_feature_vector = model(negative)

            loss = triplet_loss(
                anchor_feature_vector, positive_feature_vector, negative_feature_vector
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for data in tqdm(
                val_dataloader, total=len(val_dataloader), desc="BATCH VALIDATION "
            ):
                anchor, positive, negative = data
                anchor, positive, negative = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                )

                anchor_feature_vector = model(anchor)
                positive_feature_vector = model(positive)
                negative_feature_vector = model(negative)

                loss = triplet_loss(
                    anchor_feature_vector,
                    positive_feature_vector,
                    negative_feature_vector,
                )
                val_loss += loss.item()

                # Calculate Euclidean distances
                pos_distance = torch.norm(
                    anchor_feature_vector - positive_feature_vector, dim=1
                )
                neg_distance = torch.norm(
                    anchor_feature_vector - negative_feature_vector, dim=1
                )

                # Increment correct predictions if the positive is closer to the anchor than the negative
                correct_predictions += torch.sum(pos_distance < neg_distance).item()
                total_predictions += anchor.size(0)

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

    # For MAP@K calculations
    Ks = [1, 5, 8]
    map_at_ks = {k: 0 for k in Ks}

    # For Recall and F1 Score
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    # TODO: CLASS WISE MAP@K

    with torch.no_grad():
        for data in tqdm(
            test_dataloader, total=len(test_dataloader), desc="BATCH TEST "
        ):
            anchor, positive, negative = data
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            anchor_feature_vector = model(anchor)
            positive_feature_vector = model(positive)
            negative_feature_vector = model(negative)

            loss = triplet_loss(
                anchor_feature_vector,
                positive_feature_vector,
                negative_feature_vector,
            )
            test_loss += loss.item()

            # Compute distances
            positive_distance = (
                (anchor_feature_vector - positive_feature_vector).pow(2).sum(1)
            )
            negative_distance = (
                (anchor_feature_vector - negative_feature_vector).pow(2).sum(1)
            )

            # MAP@K calculations would require a different approach in a real scenario
            # Assuming a simplistic view where each anchor has a single positive and negative for demonstration
            distances = torch.stack([positive_distance, negative_distance], dim=1)
            sorted_distances, indices = torch.sort(distances, dim=1)

            # Recall and Precision (TP, FN, FP)
            # Here, we consider a correct prediction if positive_distance < negative_distance
            correct_preds = positive_distance < negative_distance
            true_positives += correct_preds.sum().item()
            false_negatives += (~correct_preds).sum().item()
            false_positives += (
                correct_preds.sum().item()
            )  # Simplified, usually you'd have more negatives to consider

            total_examples += anchor.size(0)

            # Simplified MAP@K for demonstration
            for k in Ks:
                correct_at_k = (indices[:, :k] == 0).any(
                    dim=1
                )  # Check if the positive is within the top K
                map_at_ks[k] += correct_at_k.float().mean().item()

    # Final MAP@K across all examples
    map_at_ks = {
        k: map_value / len(test_dataloader) for k, map_value in map_at_ks.items()
    }

    # Recall and F1 Score
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("-" * 50)
    for k, map_at_k in map_at_ks.items():
        print(f"MAP@{k}: {map_at_k:.4f}")

    print(f"Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}")
    print(f"Test Loss: {test_loss / total_examples}")

    # save the model file
    model_pt_path = os.path.join(MODEL_DIR, f"triplet_{model_name}.pt")
    torch.save(model, model_pt_path)
    print(f"model saved at {model_pt_path}")


if __name__ == "__main__":
    train_validation_test(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        "resnet50",
    )
