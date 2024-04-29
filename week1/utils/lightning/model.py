import torch
import torchvision.models as models
import pytorch_lightning as pl
from torch import Tensor
import numpy as np

from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

#from utils.mixup import mixup_criterion, mixup_data
from torchvision.transforms import v2


from torchsummary import summary
classes = [
    "coast",
    "forest",
    "highway",
    "inside_city",
    "mountain",
    "Opencountry",
    "street",
    "tallbuilding",
] 
class MITClassifier(pl.LightningModule):
    def __init__(self, num_classes=8, class_names=classes, learning_rate=None, mixup=None, mixup_alpha=None, cutmix=None, cutmix_alpha=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

        self.num_classes = num_classes
        self.class_names = class_names
        self.learning_rate = learning_rate
        self.criterion = cross_entropy
        self.metric_train = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.metric_val = MulticlassAccuracy(num_classes=num_classes, average=None)

        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix = cutmix
        self.cutmix_alpha = cutmix_alpha

        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
             'scheduler': CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=1e-6, ),
            'name': 'lr_scheduler'
        }

        return [optimizer], [lr_scheduler]


    def apply_augmentation(self, x, y):
        if self.cutmix and not self.mixup:
            operation = v2.CutMix(num_classes=self.num_classes, alpha=self.cutmix_alpha)
        elif not self.cutmix and self.mixup:
            operation = v2.MixUp(num_classes=self.num_classes, alpha=self.mixup_alpha)
        elif self.cutmix and self.mixup:
            cutmix = v2.CutMix(num_classes=self.num_classes, alpha=self.cutmix_alpha)
            mixup = v2.MixUp(num_classes=self.num_classes, alpha=self.mixup_alpha)
            operation = v2.RandomChoice([cutmix, mixup])
        elif not self.cutmix and not self.mixup: 
            def operation(x, y):
                return x, y

        return operation(x, y)

    def training_step(self, batch, batch_idx):

        x, y = batch
        x, y = self.apply_augmentation(x, y)

        logits = self(x)

        loss = self.criterion(logits, y, label_smoothing=0.1)
    
        logs = {'train_loss': loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({'loss': loss, 'logits': logits, "targets": y})
        
        return {'loss': loss, 'logits': logits, "targets": y}
    
    def on_train_epoch_end(self): 

        all_preds = torch.cat([out['logits'] for out in self.training_step_outputs])
        all_targets = torch.cat([out['targets'] for out in self.training_step_outputs])

        self.metric_train(all_preds.argmax(1), all_targets)#.argmax(1))

        scores = self.metric_train.compute()
        score_average = torch.nanmean(scores)

        print(f'Training Accuracy: {score_average}')
        logs = {'train_acc': score_average}
        for ind, score in enumerate(scores):
            logs.update({f'train_acc_class_{ind}': score})
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metric_train.reset()
        self.training_step_outputs.clear()
 


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
       
        logs = {'val_loss': loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append({'loss': loss, 'logits': logits, "targets": y})
        return {'loss': loss, 'logits': logits, "targets": y}

    def on_validation_epoch_end(self): 

        all_preds = torch.cat([out['logits'] for out in self.validation_step_outputs])
        all_targets = torch.cat([out['targets'] for out in self.validation_step_outputs])

        cf_matrix = confusion_matrix(Tensor.cpu(all_targets), Tensor.cpu(all_preds.argmax(1)), normalize='true')
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in self.class_names],
                            columns = [i for i in self.class_names])
        
        plt.figure(figsize = (12,7))       
        sn.heatmap(df_cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('cf_matrix_val.png')
        plt.close()         

        self.metric_val(all_preds.argmax(1), all_targets)
        scores = self.metric_val.compute()
        score_average = torch.nanmean(scores)
        print(f'Validation Accuracy: {score_average}')
        logs = {'val_acc': score_average}
        for ind, score in enumerate(scores):
            logs.update({f'val_acc_class_{ind}': score})
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metric_val.reset()
        self.validation_step_outputs.clear() 
