from utils.multimodal_data import *
from utils.multimodal_model import *
from utils.callbacks import *
from pytorch_lightning.loggers import CSVLogger, WandbLogger    
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import wandb
from pytorch_lightning.tuner.tuning import Tuner
import sys


def get_class_weights(labels):
    # Step 1: Compute Class Frequencies
    class_counts = np.bincount(labels)

    # Step 2: Calculate Class Weights
    total_samples = len(labels)
    class_weights = total_samples / class_counts

    # Step 3: Normalize Class Weights (Optional)
    class_weights_normalized = class_weights / np.sum(class_weights)

    return class_weights_normalized



dm = MultimodalDataModule(train_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/train_set_age_labels.csv',
                      val_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/valid_set_age_labels.csv',
                      test_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/test_set_age_labels.csv',
                      image_dir='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal',
                      batch_size=32, 
                      augmentations=True,
                      #sampler='wrs',
                      num_workers=16)
dm.setup()
train_labels = [i-1 for i in dm.train_dataset.df['AgeGroup'].values]

model = MultimodalAgePredictionModel(num_classes=7)#, class_weights=get_class_weights(train_labels))

version = 0
wandb.login()
wandb.init(project='c5_w6_multimodal', name = 'lightning')
wandb_logger = WandbLogger(name='age_prediction_logs', version=version)

# Instantiate the EarlyStopping callback
early_stop_callback = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation loss for early stopping
    patience=5,           # Number of epochs with no improvement after which training will be stopped
    verbose=True,         # Print early stopping messages
    mode='max',            # 'min' mode means training will stop when monitored quantity has stopped decreasing
)

# Instantiate the PlotMetricsCallback
plot_metrics_callback = PlotMetricsCallback(csv_path=f'task_b_logs/age_prediction_logs/version_{version}/metrics.csv', 
                                            save_dir=f'task_b_logs/age_prediction_logs/version_{version}')


# Define the learning rate monitor callback to log the learning rate
lr_monitor = LearningRateMonitor(logging_interval='epoch')


trainer = pl.Trainer(max_epochs=50, 
                     accelerator='auto', 
                     callbacks=[plot_metrics_callback, early_stop_callback, lr_monitor], 
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
