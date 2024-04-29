import lightning as L
import pytorch_lightning as pl
import torch
import torchvision
import numpy as np
import os

from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Module

class TripletNetwork(pl.LightningModule):
    def __init__(self, model=None, loss=None, optimizer=None, lr_scheduler=None):
        super().__init__()

        self.model = model
        self.criterion = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.training_losses = []
        self.validation_losses = []

    def training_step(self, batch, batch_idx):
        self.model.train()
        anchors, positives, negatives = batch

        anchor = self.model(anchors)
        positive = self.model(positives)
        negative = self.model(negatives)

        loss = self.criterion(anchor, positive, negative)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_losses.append(loss.cpu().detach().numpy())   
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_loss_epoch", np.mean(self.training_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.training_losses = []
        
    def validation_step(self, batch, batch_idx):

        self.model.eval()
        anchors, positives, negatives = batch

        anchor = self.model(anchors)
        positive = self.model(positives)
        negative = self.model(negatives)

        loss = self.criterion(anchor, positive, negative)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_losses.append(loss.cpu().detach().numpy())
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_loss_epoch", np.mean(self.validation_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_losses = []

    def test_step(self):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = {
             'scheduler': self.lr_scheduler,
            'name': 'lr_scheduler'
        }

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        return self.model(x)
    

class FasterRCNNEmbedder(Module):
    def __init__(self):
        super().__init__()

        self.backbone_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
        self.backbone = self.backbone_model.backbone
        self.backbone_layers = list(self.backbone.children())
        self.backbone_layers = Sequential(*self.backbone_layers)

        self.aap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = Sequential(Linear(1024, 1024), 
                                        ReLU(),
                                        BatchNorm1d(1024),
                                          )
        
        self.initialize_embedding_layer()
        
    def initialize_embedding_layer(self):
        # apply He initialization
        torch.nn.init.kaiming_normal_(self.embedding_layer[0].weight)
        torch.nn.init.constant_(self.embedding_layer[0].bias, 0)



    def forward(self, x):
        with torch.no_grad():
            x = self.backbone_layers(x)
        x0 = self.aap(x['0'])
        x1 = self.aap(x['1'])
        x2 = self.aap(x['2'])
        x3 = self.aap(x['3'])
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = x.view(x.size(0), -1)
        x_embedded = self.embedding_layer(x)
        return x_embedded