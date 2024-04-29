import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import WeightedRandomSampler
from collections import Counter


import torch
import torch.utils.data as data

class EqualAgeGroupSampler(data.sampler.Sampler):
    def __init__(self, dataset, num_samples=None, num_samples_per_age_group=None):
        self.dataset = dataset
        self.num_samples = num_samples
        self.num_samples_per_age_group = num_samples_per_age_group

        self.age_groups = dataset.df['AgeGroup'].unique()
        self.indices_per_age_group = {age_group: [] for age_group in self.age_groups}

        for idx in range(len(dataset)):
            age_group = dataset.df.iloc[idx]['AgeGroup'] 
            self.indices_per_age_group[age_group].append(idx)
            

        self.weights_per_age_group = {age_group: 1.0 / len(indices) for age_group, indices in self.indices_per_age_group.items()}
        print(f'Weights per age group: {self.weights_per_age_group}')

    def __iter__(self):
        indices = []
        for age_group, age_group_indices in self.indices_per_age_group.items():
            num_samples_age_group = min(self.num_samples_per_age_group, len(age_group_indices))
            indices.extend(torch.randperm(len(age_group_indices))[:num_samples_age_group])
        indices = [int(i) for i in indices]
    
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
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

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, split, augmentations=False):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.augmentations = augmentations

        # Define Albumentations augmentation pipeline if required
        if self.augmentations:
            self.augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.5),
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.ChannelShuffle(p=0.5),
                A.ToGray(p=0.5),
                A.ChannelDropout(p=0.5),

            ])
        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        age_group = self.df.iloc[idx, 2] # Assuming AgeGroup is the third column
        img_name = self.df.iloc[idx, 0].replace('mp4', 'jpg') # Assuming VideoName is the first column
        img_path = os.path.join(self.root_dir, self.split, str(age_group), img_name)  
        image = Image.open(img_path)
        
        # Convert PIL image to NumPy array
        image = np.array(image)
        
        # Apply Albumentations augmentations if required
        if self.augmentations:
            augmented = self.augmentation_pipeline(image=image)
            image = augmented['image']
        
        # Apply generic preprocessing transform
        image = self.transform(image)
        
        age_group = torch.tensor(age_group-1)  # Assuming AgeGroup is the third column
        gender = torch.tensor(self.df.iloc[idx, 3]-1)
        ethnicity = torch.tensor(self.df.iloc[idx, 4]-1)

        return image, age_group, gender, ethnicity

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_csv_file, val_csv_file, test_csv_file, image_dir, batch_size=32, augmentations=False, num_workers=4, sampler=False):

        super().__init__()
        self.train_csv_file = train_csv_file
        self.val_csv_file = val_csv_file
        self.test_csv_file = test_csv_file

        self.image_dir = image_dir
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.num_workers = num_workers
        self.sampler = sampler
        self.save_hyperparameters()


    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            csv_file=self.train_csv_file,
            root_dir=self.image_dir,
            split='train',
            augmentations=self.augmentations
        )
        self.val_dataset = CustomDataset(
            csv_file=self.val_csv_file,
            root_dir=self.image_dir,
            split='valid',
            augmentations=None
        )
        self.test_dataset = CustomDataset(
            csv_file=self.test_csv_file,
            root_dir=self.image_dir,
            split='test',
            augmentations=None
        )

    def train_dataloader(self):
        train_labels = [label - 1 for label in self.train_dataset.df['AgeGroup'].tolist()]
        sampler, shuffle = define_sampler(train_labels, self.sampler)
        # if self.sampler:
            # sampler = EqualAgeGroupSampler(self.train_dataset, num_samples=len(self.train_dataset), num_samples_per_age_group=self.batch_size//7)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    

