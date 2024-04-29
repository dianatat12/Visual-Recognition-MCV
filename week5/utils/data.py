from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json
import os
import cv2
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import transformers

from .transforms import albumentations_transform, preprocess


class ImageTextRetrievalDataset(Dataset):
    def __init__(self, json, img_files_root, split, transform, dims=(448, 448)):

        self.json = json
        self.img_directory = os.path.join(img_files_root)
        self.split = split
        self.transform = transform
        self.preprocess = preprocess(dims)

        self.prepare_data()

    def match_items(self, list1, list2):
        # Create a dictionary from list2 with 'image_id' as the key
        dict1 = {item["id"]: item for item in list1}
        # Iterate over list1 and add the corresponding item from list2
        for item in list2:
            if item["image_id"] in dict1:
                item.update(dict1[item["image_id"]])

        return list2

    def read_captions_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        image_data = data["images"]
        caption_data = data["annotations"]
        return image_data, caption_data

    def prepare_data(self):
        image_data, caption_data = self.read_captions_json(self.json)
        print(f"Image data rows: {len(image_data)}")
        print(f"Caption data rows: {len(caption_data)}")
        data_list = self.match_items(image_data, caption_data)
        df = pd.DataFrame(data_list)
        print(f"Dataframe length: {len(df)}")
        self.df = df.drop(
            columns=[
                "license",
                "coco_url",
                "flickr_url",
                "date_captured",
                "height",
                "width",
                "id",
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_file = self.df["file_name"][index]
        img_path = os.path.join(self.img_directory, img_file)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = np.array(img)
            augmented = self.transform(image=img)
            img = augmented["image"]

        img = self.preprocess(img)

        text = self.df["caption"][index]

        return img, text


import torch.nn.functional as F


class ImageTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers=4,
        img_embeddings_directory=None,
        text_embeddings_directory=None,
        transform=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_embeddings_directory = img_embeddings_directory
        self.text_embeddings_directory = text_embeddings_directory
        self.transform = transform
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            f"bert-base-uncased"
        )

    def collate_fn(self, batch):

        images = []
        texts = []
        for pair in batch:
            # Get the maximum length of the sequences in the batch
            image, text = pair
            images.append(image)
            texts.append(text)

        text_tokenized = self.tokenizer(
            texts, return_tensors="pt", padding=True, max_length=25, truncation=True
        )["input_ids"]
        images = torch.stack(images)
        return images, text_tokenized, texts

    def train_dataloader(self):
        train_dataset = ImageTextRetrievalDataset(
            json=f"/ghome/group04/mcv/datasets/C5/COCO/captions_train2014.json",
            img_files_root="/ghome/group04/mcv/datasets/C5/COCO/train2014",
            transform=self.transform,
            split="train",
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        val_dataset = ImageTextRetrievalDataset(
            json=f"/ghome/group04/mcv/datasets/C5/COCO/captions_val2014.json",
            img_files_root="/ghome/group04/mcv/datasets/C5/COCO/val2014",
            transform=None,
            split="val",
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    dm = ImageTextDataModule(batch_size=16)
    train_dl = dm.train_dataloader()
    print(len(train_dl))
