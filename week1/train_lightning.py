import torch
import wandb 
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import (
    LearningRateFinder,
    BatchSizeFinder,
    RichModelSummary,
)
from pytorch_lightning.loggers import WandbLogger

from lightning.pytorch.tuner import Tuner

import torch.nn.init as init

import matplotlib.pyplot as plt

plt.set_cmap("cividis")
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg", "pdf")  # For export
import matplotlib

matplotlib.rcParams["lines.linewidth"] = 2.0

from utils.lightning.data import MITDataModule
from utils.lightning.transforms import albumentations_transform
from utils.lightning.model import MITClassifier

wandb.login()
wandb.init(project='c5_w1', name = 'lightning')

BATCH_SIZE = 128
DATA_ROOT = "/ghome/group04/augmented_dataset/"
IMG_SIZE = 256
EPOCHS = 300
GPUS = 0
CLASSES = [
    "coast",
    "forest",
    "highway",
    "inside_city",
    "mountain",
    "Opencountry",
    "street",
    "tallbuilding",
]
NUM_CLASSES = len(CLASSES)

LEARNING_RATE = 0.0004
MIXUP = False
MIXUP_ALPHA = 0.2

CUTMIX = False
CUTMIX_ALPHA = 0.2

dm = MITDataModule(
    batch_size=BATCH_SIZE,
    root_dir=DATA_ROOT,
    dims=(IMG_SIZE, IMG_SIZE),
    #transforms=albumentations_transform(),
    sampler="wrs",
    num_workers=2,
)

model = MITClassifier(
    learning_rate=LEARNING_RATE,
    num_classes=NUM_CLASSES,
    class_names=CLASSES,
    mixup=MIXUP,
    mixup_alpha=MIXUP_ALPHA,
    cutmix=CUTMIX,
    cutmix_alpha=CUTMIX_ALPHA,
)

# checkpointer
checkpointer = ModelCheckpoint(
    monitor="val_acc", save_top_k=1, mode="max", save_weights_only=True
)

# lr_monitor
learningrate_monitor = LearningRateMonitor(logging_interval="step")

# early stopping
early_stopping = EarlyStopping(
    monitor=("val_acc"), min_delta=0.00, patience=10, mode="max"
)


# Run learning rate finder
lr_finder = LearningRateFinder(
    min_lr=0.00001,
    max_lr=0.01,
    num_training_steps=100,
    mode="exponential",
    early_stop_threshold=4.0,
    update_attr=True,
    attr_name="learning_rate",
)


rich_summary = RichModelSummary(max_depth=20)

wandb_logger = WandbLogger()

# # training
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    devices=1,
    accelerator="gpu",
    num_nodes=1,
    precision=32,
    log_every_n_steps=1,
    logger=wandb_logger,
    callbacks=[
        lr_finder,
        checkpointer,
        learningrate_monitor,
        early_stopping,
        rich_summary,
    ],
    num_sanity_val_steps=0,
)

trainer.fit(model=model, datamodule=dm)
