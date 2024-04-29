import faiss
from model import ImageTextRetrievalModel
from data import ImageTextDataModule
import torch
import numpy as np
import json
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    average_precision_score,
)


def get_img_file_name(img_id, set):
    return "COCO_{}2014_{:012d}.jpg".format(set, img_id)


def prepare_data(json_file, mode):
    with open(json_file, "r") as file:
        # Load the JSON data
        data = json.load(file)[mode]
    print(f"Loaded {len(data)} classes from {json_file}")
    img_ids = []
    labels = []
    # loop over classes
    for key in tqdm(data.keys(), desc=f"Preparing {mode} data"):
        class_ = key
        images_with_class = data[key]
        # loop over images with the class
        for image_id in images_with_class:
            # if it's a new image, add it to the list of images and create a label list for it
            if image_id not in img_ids:
                img_ids.append(image_id)
                labels.append([])
        # loop over images and add the class to the label list if it's in the list of images
        for i, img_id in enumerate(img_ids):
            if img_id in images_with_class:
                labels[i].append(int(class_))

    data_split = "train" if mode in ["train", "database"] else "val"

    img_files = [get_img_file_name(img_id, data_split) for img_id in img_ids]
    return img_files, labels


data_json = "/home/georg/projects/university/C5/task3/dataset/COCO/mcv_image_retrieval_annotations.json"
# extract embeddings from the training images
train_imgs_path = "/home/georg/projects/university/C5/task3/dataset/COCO/train2014"
train_img_files, train_labels = prepare_data(json_file=data_json, mode="database")

val_imgs_path = "/home/georg/projects/university/C5/task3/dataset/COCO/val2014"
val_img_files, val_labels = prepare_data(json_file=data_json, mode="test")

# Load the model
model = ImageTextRetrievalModel.load_from_checkpoint(
    mode="img2text",
    checkpoint_path="/path/to/img2text/weights.ckpt",
)
model.eval()

# Load the data
dm = ImageTextDataModule(batch_size=64, num_workers=4)

train_dl = dm.train_dataloader()
train_dl.transform = None
val_dl = dm.val_dataloader()


train_image_embeddings = []
train_text_embeddings = []
for i, (img, text) in enumerate(train_dl):
    train_image_embedding, train_text_embedding = model.forward(img, text)
    train_image_embeddings.append(train_image_embedding)
    train_text_embeddings.append(train_text_embedding)

train_image_embeddings = np.array(torch.cat(train_image_embeddings).detach().cpu())
train_text_embeddings = np.array(torch.cat(train_text_embeddings).detach().cpu())
# Create the index
faiss.normalize_L2(train_text_embeddings)

index = faiss.IndexFlatL2(2048)
index.add(train_text_embeddings)
sums = 0
for i, (img, text) in enumerate(val_dl):
    image_embeddings, text_embeddings = model.forward(img, text)
    D, I = index.search(np.array(image_embeddings.detach().cpu()), 5)


k = 5
visualize = False

targets = []
preds = []
for i, retrieval_indices in enumerate(I):
    query_img_file = val_img_files[i]
    query_img_path = os.path.join(val_imgs_path, query_img_file)
    query_img_labels = val_labels[i]
    retrieved_image_files = []
    retrieved_image_labels = []
    targets.append(1)

    for train_idx in retrieval_indices[:k]:
        retrieved_image_files.append(train_img_files[train_idx])
        retrieved_image_paths = [
            os.path.join(train_imgs_path, file) for file in retrieved_image_files
        ]
        retrieved_image_labels.extend(train_labels[train_idx])

    preds.append(
        len(set(query_img_labels).intersection(set(retrieved_image_labels))) > 0
    )
    if visualize == True:
        query_img = cv2.imread(query_img_path)
        query_img = cv2.resize(query_img, (224, 224))
        retrieved_imgs = [cv2.imread(file) for file in retrieved_image_paths]
        retrieved_imgs = [cv2.resize(img, (224, 224)) for img in retrieved_imgs]

        fig, ax = plt.subplots(1, 6, figsize=(18, 3))
        ax[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Query Image")
        for j, img in enumerate(retrieved_imgs):
            ax[j + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[j + 1].set_title(f"Retrieved Image {j+1}")
        plt.savefig("query_plots/{:03d}.png".format(i))
        plt.close()

print(len(targets))
print(len(preds))
precision = precision_score(targets, preds, average="binary")
recall = recall_score(targets, preds, average="binary")
accuracy = accuracy_score(targets, preds)
f1 = 2 * (precision * recall) / (precision + recall)
print(f"Precision: {precision}, \nRecall: {recall}, \nAccuracy: {accuracy}, \nF1: {f1}")
average_precision = average_precision_score(targets, preds)
print(f"Average precision: {average_precision}")
