import torch
import torchvision.models as models
import transformers
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import pytorch_lightning as pl
import torch.nn.functional as F


class ImageTextRetrievalModel(pl.LightningModule):
    def __init__(
        self,
        text_embedding_dim=768,
        image_embedding_dim=2048,
        margin=0.2,
        pretrained_bert="bert-base-uncased",
        mode="text2img",
        learning_rate=1e-3,
    ):
        super().__init__()

        # Load pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the classification layer

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_bert)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert)

        # Projection layers to map image and text embeddings to the same dimension
        self.image_projection = nn.Linear(image_embedding_dim, image_embedding_dim)
        self.text_projection = nn.Linear(text_embedding_dim, image_embedding_dim)
        # initialize weights using he initialization for projection layers
        nn.init.kaiming_normal_(self.image_projection.weight)
        nn.init.kaiming_normal_(self.text_projection.weight)

        # Triplet loss margin
        self.margin = margin

        self.mode = mode
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=0.0001, last_epoch=-1
        )

    def forward(self, images, texts):
        # Forward pass for images
        image_features = self.resnet(images)
        image_embeddings = self.image_projection(image_features)

        text_outputs = self.bert(input_ids=texts)
        text_embeddings = self.text_projection(
            text_outputs.last_hidden_state[:, 0, :]
        )  # Use CLS token
        return image_embeddings, text_embeddings

    def triplet_loss(self, anchor, positive, negative):
        loss = torch.nn.TripletMarginLoss(
            margin=self.margin,
            p=2.0,
            eps=1e-06,
            swap=False,
            size_average=None,
            reduce=None,
            reduction="mean",
        )
        return loss(anchor, positive, negative)

    def training_step(self, batch, batch_idx):
        images, texts = batch

        # Forward pass to get anchor embeddings
        anchor_image_embeddings, anchor_text_embeddings = self(images, texts)

        positive_image_embeddings, positive_text_embeddings = (
            anchor_image_embeddings,
            anchor_text_embeddings,
        )

        negative_image_embeddings, negative_text_embeddings = self.mine_hard_negatives(
            anchor_image_embeddings, anchor_text_embeddings
        )

        if self.mode == "text2img":
            # Concatenate all embeddings

            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_text_embeddings,
                positive_image_embeddings,
                negative_image_embeddings,
            )

        elif self.mode == "img2text":

            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_image_embeddings,
                positive_text_embeddings,
                negative_text_embeddings,
            )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        images, texts = batch

        # Forward pass to get anchor embeddings
        anchor_image_embeddings, anchor_text_embeddings = self(images, texts)

        positive_image_embeddings, positive_text_embeddings = (
            anchor_image_embeddings,
            anchor_text_embeddings,
        )

        negative_image_embeddings, negative_text_embeddings = self.mine_hard_negatives(
            anchor_image_embeddings, anchor_text_embeddings
        )

        if self.mode == "text2img":
            # Concatenate all embeddings

            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_text_embeddings,
                positive_image_embeddings,
                negative_image_embeddings,
            )

        elif self.mode == "img2text":

            # Compute triplet loss
            loss = self.triplet_loss(
                anchor_image_embeddings,
                positive_text_embeddings,
                negative_text_embeddings,
            )
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",  # adjust the learning rate scheduler's step interval
                "monitor": "val_loss",  # monitor a metric to adjust the learning rate
            },
        }

    def mine_hard_negatives(self, image_embeddings, text_embeddings):
        # find the hardest negative samples
        # Compute cosine similarity between anchor and all other samples
        image_similarity = F.cosine_similarity(
            image_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=2
        )
        text_similarity = F.cosine_similarity(
            text_embeddings.unsqueeze(1), image_embeddings.unsqueeze(0), dim=2
        )

        # Find the hardest negative samples
        hard_negative_image_embeddings = text_embeddings[image_similarity.argmax(dim=1)]
        hard_negative_text_embeddings = image_embeddings[text_similarity.argmax(dim=1)]

        return hard_negative_image_embeddings, hard_negative_text_embeddings
