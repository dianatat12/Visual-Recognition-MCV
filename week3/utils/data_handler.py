import os
import random
from glob import glob


from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TripletDataset(Dataset):
    def __init__(self, image_dir: str, image_extension: str = "jpg", transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_extension = image_extension
        self.class_to_images = {}
        self.all_images = []

        for class_dir in os.listdir(image_dir):
            class_path = os.path.join(image_dir, class_dir)
            if os.path.isdir(class_path):
                images = glob(os.path.join(class_path, f"*.{image_extension}"))
                if images:
                    self.class_to_images[class_dir] = images[:1000]
                    self.all_images.extend(images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_path = self.all_images[idx]
        anchor_class = os.path.basename(os.path.dirname(anchor_path))

        positive_path = random.choice(self.class_to_images[anchor_class])

        while positive_path == anchor_path:
            positive_path = random.choice(self.class_to_images[anchor_class])

        negative_class = random.choice(list(self.class_to_images.keys()))
        while negative_class == anchor_class:
            negative_class = random.choice(list(self.class_to_images.keys()))

        # random negative
        negative_path = random.choice(self.class_to_images[negative_class])

        # read images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transforms:
            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

        return anchor, positive, negative


def get_dataloader(
    train_dir: str,
    test_dir: str,
    val_dir: str,
    train_batch_size: int,
    test_val_batch_size: int,
    num_workers: int,
    transforms,
):
    # dataset
    train_dataset = TripletDataset(image_dir=train_dir, transforms=transforms)
    val_dataset = TripletDataset(image_dir=val_dir, transforms=transforms)
    test_dataset = TripletDataset(image_dir=test_dir, transforms=transforms)

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
