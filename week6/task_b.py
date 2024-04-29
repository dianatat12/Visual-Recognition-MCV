from utils.data import *
from utils.model import *
from utils.callbacks import *
from pytorch_lightning.loggers import CSVLogger, WandbLogger    
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint







dm = CustomDataModule(train_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/train_set_age_labels.csv',
                      val_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/valid_set_age_labels.csv',
                      test_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/test_set_age_labels.csv',
                      image_dir='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal',
                      batch_size=256, 
                      augmentations=True,
                      #sampler='wrs',
                      num_workers=8)

model = AgePredictionModel(num_classes=7)

version = 0
# Instantiate the CSVLogger callback
#csv_logger = CSVLogger(save_dir='task_b_logs', name='age_prediction_logs', version=version)
wandb.login()
wandb.init(project='c5_w6', name = 'lightning')
wandb_logger = WandbLogger(name='age_prediction_logs', version=version)

# Instantiate the EarlyStopping callback
early_stop_callback = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation loss for early stopping
    patience=10,           # Number of epochs with no improvement after which training will be stopped
    verbose=True,         # Print early stopping messages
    mode='max',            # 'min' mode means training will stop when monitored quantity has stopped decreasing
)

# Instantiate the PlotMetricsCallback
plot_metrics_callback = PlotMetricsCallback(csv_path=f'task_b_logs/age_prediction_logs/version_{version}/metrics.csv', 
                                            save_dir=f'task_b_logs/age_prediction_logs/version_{version}')

scheduler = CosineAnnealingLR(model.optimizer, T_max=50)  # Adjust T_max (number of epochs) as needed

# Define the learning rate monitor callback to log the learning rate
lr_monitor = LearningRateMonitor(logging_interval='epoch')


# Define the directory to save the checkpoints
checkpoint_dir = 'checkpoints/'

# Instantiate the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,  # Directory to save the checkpoints
    filename='model-{epoch:02d}-{val_loss:.2f}',  # File name format for saving the checkpoints
    monitor='val_loss',  # Monitor validation loss for saving the best model
    mode='min',  # 'min' mode saves the model with the minimum monitored quantity (e.g., validation loss)
    save_top_k=3,  # Save the top 3 best models
    verbose=True  # Print verbose messages
)


trainer = pl.Trainer(max_epochs=50, 
                     accelerator='auto', 
                     callbacks=[plot_metrics_callback, early_stop_callback, lr_monitor,checkpoint_callback], 
                     logger=wandb_logger)

tuner = Tuner(trainer)

# Run learning rate finder
lr_finder = tuner.lr_find(model, dm)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.savefig('lr_finder_plot.png')

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

# update hparams of the model
model.hparams.lr = new_lr

trainer.fit(model, dm)
trainer.test(model, datamodule=dm)
