import torch
import cv2
import pytorch_lightning as pl
import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from .transforms import preprocess
from torch.utils.data import WeightedRandomSampler


class MITDataset(Dataset):

    def __init__(self, dir, dims, transforms=None):

        self.dir = dir
        self.transform = transforms
        self.dims = dims
        self.images, self.labels, self.classes = self.get_images_and_labels()

    def get_images_and_labels(self):
        images = []
        labels = []
        classes = []
        class_dir_list = sorted(os.listdir(self.dir), key=lambda x: x.lower())
        for label, name in enumerate(class_dir_list):
            classes.append(name)
            class_directory = os.path.join(self.dir, name)
            for image_file in os.listdir(class_directory):

                images.append(image_file)
                labels.append(label)
        # Combine the lists using zip
        combined_lists = list(zip(images, labels))
        # Shuffle the combined list
        random.shuffle(combined_lists)
        # Unzip the shuffled list
        images, labels = zip(*combined_lists)
        return images, labels, classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_file, label = self.images[idx], self.labels[idx]
        image_path = os.path.join(self.dir, self.classes[label], image_file)

        # read image with OpenCV and convert to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # preprocess images
        preprocesser = preprocess(self.dims)
        preprocessed = preprocesser(image=image)
        image = preprocessed["image"]
        return image, int(label)


class MITDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        root_dir,
        dims=(224, 224),
        transforms=None,
        sampler=None,
        num_workers=8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "test")

        self.dims = dims
        self.transforms = transforms
        self.num_workers = num_workers
        self.sampler = sampler

    def setup(self, stage="fit"):
        self.train_dataset = MITDataset(
            dir=self.train_dir, transforms=self.transforms, dims=self.dims
        )
        self.val_dataset = MITDataset(dir=self.val_dir, transforms=None, dims=self.dims)

    def train_dataloader(self):
        sampler, shuffle = define_sampler(self.train_dataset.labels, self.sampler)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def get_class_weights(labels):

    weight = torch.tensor([1 / np.sum(labels == i) for i in np.unique(labels)])
    weight = weight / weight.sum()
    weight = weight[torch.tensor(list(labels))]
    return weight


def define_sampler(labels, sampler):
    sampler_ = None
    shuffle_ = True
    if sampler == "wrs":
        weights = get_class_weights(labels)
        sampler_ = WeightedRandomSampler(weights, len(weights))
        shuffle_ = False
    return sampler_, shuffle_
