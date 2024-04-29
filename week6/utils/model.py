import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from collections import Counter


class AgePredictionModel(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = nn.Identity()

        self.lr = lr
        self.configure_optimizers()

    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        pred_labels = torch.argmax(y_hat, dim=1)
        accuracy = (pred_labels == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, gender, ethnicity = batch
        y_hat = self(x)
        val_loss = nn.CrossEntropyLoss()(y_hat, y)
        pred_labels = torch.argmax(y_hat, dim=1)
        accuracy = (pred_labels == y).float().mean()
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y, gender, ethnicity = batch
        y_hat = self(x)
        test_loss = nn.CrossEntropyLoss()(y_hat, y)
        pred_labels = torch.argmax(y_hat, dim=1)
        accuracy = (pred_labels == y).float().mean()

        # Calculate overall accuracy
        self.log('test_accuracy', accuracy.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss', test_loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        # Calculate accuracies per age group, ethnicity, and gender
        age_groups = y
        ethnicities = ethnicity
        genders = gender

        for age_group in age_groups.unique():
            age_mask = (age_groups == age_group)
            age_accuracy = (pred_labels[age_mask] == y[age_mask]).float().mean()
            self.log(f'test_accuracy_age_group_{age_group}', age_accuracy.item())

        for ethnicity_val in ethnicities.unique():
            ethnicity_mask = (ethnicities == ethnicity_val)
            ethnicity_accuracy = (pred_labels[ethnicity_mask] == y[ethnicity_mask]).float().mean()
            self.log(f'test_accuracy_ethnicity_{ethnicity_val}', ethnicity_accuracy.item())

        for gender_val in genders.unique():
            gender_mask = (genders == gender_val)
            gender_accuracy = (pred_labels[gender_mask] == y[gender_mask]).float().mean()
            self.log(f'test_accuracy_gender_{gender_val}', gender_accuracy.item())

        return test_loss
    
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer