from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json
import os
import cv2
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

from .transforms import albumentations_transform, preprocess

def get_img_file_name(img_id, set):
    return 'COCO_{}2014_{:012d}.jpg'.format(set, img_id)

class TripletDataset(Dataset):
    def __init__(self, data_dir, mode, json_file, transforms=None, dims=(224, 224)):
        self.data_dir = data_dir
        self.mode = mode 
        self.preprocess = preprocess(dims)
        self.transforms = transforms
        # Open the JSON file
        with open(json_file, 'r') as file:
            # Load the JSON data
            self.data = json.load(file)[self.mode]

        self.prepare_data()

    def prepare_data(self):
        self.img_ids = []
        self.labels = []
        # loop over classes 
        for key in tqdm(self.data.keys(), desc=f'Preparing {self.mode} data'):
            class_ = key
            images_with_class = self.data[key]
            # loop over images with the class
            for image_id in images_with_class:
                # if it's a new image, add it to the list of images and create a label list for it
                if image_id not in self.img_ids:
                    self.img_ids.append(image_id)
                    self.labels.append([])
            # loop over images and add the class to the label list if it's in the list of images
            for i, img_id in enumerate(self.img_ids):
                if img_id in images_with_class:
                    self.labels[i].append(int(class_))

        self.data_split = 'train' if self.mode == 'train' else 'val'

        self.img_files = [get_img_file_name(img_id, self.data_split) for img_id in self.img_ids]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
            
        anchor_file = self.img_files[index]
        positive_file, negative_file = self.get_pos_neg_samples(index)

        anchor = self.load_and_preprocess_image(anchor_file)
        positive = self.load_and_preprocess_image(positive_file)
        negative = self.load_and_preprocess_image(negative_file)
    
        return anchor, positive, negative
        

    def get_pos_neg_samples(self, index):
        anchor_label = self.labels[index]
        max_matches = 0
        min_matches = 1000
        positive_indices = None
        negative_indices = None
        
        for i, label in enumerate(self.labels):

            num_matches = len(set(anchor_label) & set(label))
            if num_matches > max_matches and i != index:
                max_matches = num_matches
                positive_indices = [i]
            if num_matches == max_matches and num_matches != 0 and i != index:
                positive_indices.append(i)

            if num_matches < min_matches and i != index:
                min_matches = num_matches
                negative_indices = [i]
            if num_matches == min_matches and i != index:
                negative_indices.append(i)

        positive_index = random.choice(positive_indices)
        negative_index = random.choice(negative_indices)

        positive = self.img_files[positive_index]
        negative = self.img_files[negative_index]

        return positive, negative
    
    def load_and_preprocess_image(self, file):
        image_path = os.path.join(self.data_dir, self.data_split + '2014', file)
        #image = cv2.imread(image_path)
        image = Image.open(image_path).convert('RGB')

        if self.transforms and self.mode == 'train':
            image = np.array(image)
            augmented = self.transforms(image=image)
            image = augmented["image"]

        preprocessed = self.preprocess(image)

        return preprocessed
           
class TripletDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, json_file, transforms=None, dims=(224, 224), num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.json_file = json_file
        self.transforms = transforms
        self.dims = dims
        self.num_workers = num_workers

    def train_dataloader(self):
        train_dataset = TripletDataset(self.data_dir, 'train', self.json_file, self.transforms, self.dims)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        val_dataset = TripletDataset(self.data_dir, 'val', self.json_file, self.transforms, self.dims)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        test_dataset = TripletDataset(self.data_dir, 'test', self.json_file, self.transforms, self.dims)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
