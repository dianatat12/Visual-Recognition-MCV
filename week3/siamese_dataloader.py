import os
import random
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class SiameseDataset(Dataset):

    def __init__(self, image_dir: str, image_extension: str = "jpg", transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_extension = image_extension
        self.class_to_images = {}
        self.all_images = []

        for class_dir in os.listdir(image_dir):
            class_path = os.path.join(image_dir, class_dir)
            if os.path.isdir(class_path):
                images = glob(os.path.join(class_path, f"*.{image_extension}"))
                if images:
                    self.class_to_images[class_dir] = images
                    self.all_images.extend(images)

    def __getitem__(self, index):
        img1_path = self.all_images[index]
        img1_label = os.path.basename(os.path.dirname(img1_path))

        siamese_target = random.randint(0, 1)

        if siamese_target == 1:
            # Select an image from the same class
            siamese_index = index
            while siamese_index == index:
                siamese_index = random.randint(
                    0, len(self.class_to_images[img1_label]) - 1
                )
            img2_path = self.class_to_images[img1_label][siamese_index]
        else:
            # Select an image from a different class
            classes = list(self.class_to_images.keys())
            classes.remove(img1_label)
            siamese_label = random.choice(classes)
            img2_path = random.choice(self.class_to_images[siamese_label])

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, siamese_target

    def __len__(self):
        return len(self.all_images)


def get_dataloader(
    train_dir: str,
    test_dir: str,
    val_dir: str,
    train_batch_size: int,
    test_val_batch_size: int,
    num_workers: int,
    transform,
):
    # dataset
    train_dataset = SiameseDataset(image_dir=train_dir, transform=transform)
    val_dataset = SiameseDataset(image_dir=val_dir, transform=transform)
    test_dataset = SiameseDataset(image_dir=test_dir, transform=transform)

    # dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=test_val_batch_size,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=test_val_batch_size,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader, val_dataloader
