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
from transformers import BertModel
from transformers import Wav2Vec2Processor
from utils.audeering_audio import AgeGenderModel
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingLR


class MultimodalAgePredictionModel(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3, class_weights=None):
        super().__init__()
        #load class weights to gpu
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda')

        self.img_embedding_dim = 2048
        self.text_embedding_dim = 768
        self.audio_embedding_dim = 1024
        self.resnet = resnet50(pretrained=True)
        #self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = nn.Identity()

        self.image_projection = nn.Sequential(
            nn.BatchNorm1d(self.img_embedding_dim),
            nn.Linear(self.img_embedding_dim, self.img_embedding_dim),
            nn.ReLU(),
        )

        self.text_feature_extractor = BertModel.from_pretrained('bert-base-uncased')

        self.text_projection = nn.Sequential(
            nn.BatchNorm1d(self.text_embedding_dim),
            nn.Linear(self.text_embedding_dim, self.img_embedding_dim),
            nn.ReLU(),
        )

        self.audio_model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
        self.audio_processor = Wav2Vec2Processor.from_pretrained(self.audio_model_name)
        self.audio_feature_extractor = AgeGenderModel.from_pretrained(self.audio_model_name)

        self.audio_projection = nn.Sequential(
            nn.BatchNorm1d(self.audio_embedding_dim),
            nn.Linear(self.audio_embedding_dim, self.img_embedding_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(nn.Linear(self.img_embedding_dim * 3, self.img_embedding_dim),
                                nn.ReLU(),
                                nn.Linear(self.img_embedding_dim, num_classes))

        self.initialize_weights(self.image_projection)
        self.initialize_weights(self.text_projection)
        self.initialize_weights(self.audio_projection)
        self.initialize_weights(self.fc)


        self.lr = lr
        self.configure_optimizers()


    def initialize_weights(self, layer):
        for m in layer:
            if isinstance(m, nn.Linear):
                # He initialization for Linear layers
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)  

    def extract_text_features(self, x: str) -> np.ndarray:
        r"""Extract embeddings from text."""
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        with torch.no_grad():
            x = self.text_feature_extractor(input_ids=input_ids.squeeze(dim=1), attention_mask=attention_mask.squeeze(dim=1))
            last_hidden_state = x.last_hidden_state
            # average pooling of embeddings
            embedding = torch.mean(last_hidden_state, dim=1).squeeze()
            
        return embedding

    def extract_audio_features(
            self,
            x: np.ndarray,
            sampling_rate: int = 16000,
            embeddings: bool = True,
        ) -> np.ndarray:
        r"""Predict age and gender or extract embeddings from raw audio signal."""

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.audio_processor(x, sampling_rate=sampling_rate)
        y = y['input_values'][0]
        #y = y.reshape(1, -1)
        #print(y.shape)

        y = torch.from_numpy(y).to('cuda').squeeze(1).squeeze(1)

        # run through model
        with torch.no_grad():
            y = self.audio_feature_extractor(y)
            if embeddings:
                y = y[0]
            else:
                y = torch.hstack([y[1], y[2]])

        # convert to numpy
        #y = y.detach().cpu().numpy()

        return y

    def extract_image_features(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        return x#self.fc(x)
    
    def forward(self, x):
        image, text, audio = x
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(text)
        audio_features = self.extract_audio_features(audio)
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)
        audio_features = self.audio_projection(audio_features)
        # concatenate the features
        x = torch.cat((image_features, text_features, audio_features), dim=1)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        image, y, _, _, transcription, audio = batch
        y_hat = self([image, transcription, audio])
        if self.class_weights is None:
            loss = nn.CrossEntropyLoss()(y_hat, y)
        else:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(y_hat, y)
        pred_labels = torch.argmax(y_hat, dim=1)
        accuracy = (pred_labels == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, y, _, _, transcription, audio = batch
        y_hat = self([image, transcription, audio])
        val_loss = nn.CrossEntropyLoss()(y_hat, y)
        pred_labels = torch.argmax(y_hat, dim=1)
        accuracy = (pred_labels == y).float().mean()
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        image, y, gender, ethnicity, transcription, audio = batch
        y_hat = self([image, transcription, audio])
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
        #scheduler = CosineAnnealingLR(self.optimizer, T_max=50)  # Adjust T_max (number of epochs) as needed

        return {'optimizer': self.optimizer}