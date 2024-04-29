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
import pickle
from transformers import BertTokenizer
import torchaudio
from transformers import Wav2Vec2Processor

import torch
import torch.utils.data as data
    
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

class MultimodalDataset(Dataset):
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

        # load pre-trained BERT model and tokenizer
        self.text_model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.text_model_name)
        
    def __len__(self):
        return len(self.df)
    
    def text_loader(self, text_path):
        with open(text_path, 'rb') as f:
            transcription = pickle.load(f)
        transcription = transcription.lower()
        transcription = ''.join(e for e in transcription if e.isalnum() or e.isspace())
        return transcription

    def __getitem__(self, idx):
        age_group = self.df.iloc[idx, 2] # Assuming AgeGroup is the third column

        img_name = self.df.iloc[idx, 0].replace('mp4', 'jpg') # Assuming VideoName is the first column
        img_path = os.path.join(self.root_dir, self.split, str(age_group), img_name)  
        text_name = self.df.iloc[idx, 0].replace('mp4', 'pkl')
        text_path = os.path.join(self.root_dir, self.split, str(age_group), text_name)  
        audio_name = self.df.iloc[idx, 0].replace('mp4', 'wav')
        audio_path = os.path.join(self.root_dir, self.split, str(age_group), audio_name)  

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

        transcription = self.text_loader(text_path)
        transcription_tokenized = self.tokenizer(transcription, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        if audio.shape[1] != 244832:
            audio = torch.nn.functional.pad(audio, (0, 244832 - audio.shape[1]))

        return image, age_group, gender, ethnicity, transcription_tokenized, audio

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, train_csv_file, val_csv_file, test_csv_file, image_dir, batch_size=32, augmentations=False, num_workers=4, sampler=None):

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
        self.train_dataset = MultimodalDataset(
            csv_file=self.train_csv_file,
            root_dir=self.image_dir,
            split='train',
            augmentations=self.augmentations
        )
        self.val_dataset = MultimodalDataset(
            csv_file=self.val_csv_file,
            root_dir=self.image_dir,
            split='valid',
            augmentations=None
        )
        self.test_dataset = MultimodalDataset(
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
    

