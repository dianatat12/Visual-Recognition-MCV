from utils.multimodal_data import *
from utils.multimodal_model import *
from utils.callbacks import *
from pytorch_lightning.loggers import CSVLogger, WandbLogger    
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pytorch_lightning.tuner.tuning import Tuner




dm = MultimodalDataModule(train_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/train_set_age_labels.csv',
                            val_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/valid_set_age_labels.csv',
                            test_csv_file='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/test_set_age_labels.csv',
                            image_dir='/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal',
                            batch_size=4, 
                            augmentations=True,
                            sampler='wrs',
                            num_workers=8)

model = MultimodalAgePredictionModel.load_from_checkpoint(checkpoint_path="/home/georg/projects/university/C5/MCV-C5-G4/week6/lightning_logs/frqr6syt/checkpoints/epoch=24-step=600.ckpt", num_classes=7)

dm.setup()
test_dl = dm.test_dataloader()
preds = []
for batch in test_dl:
    images, labels, _,_, transcription, audio = batch
    y_hat = model(images.to('cuda'), transcription.to('cuda'), audio.to('cuda')).detach().cpu().numpy()
    preds.extend([np.argmax(result) for result in y_hat])

gt_df = pd.read_csv('/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/test_set_age_labels.csv')

results_df = pd.DataFrame({'VideoName':gt_df['VideoName'],'ground_truth':gt_df['AgeGroup'],'prediction':preds})

print(results_df)
results_df.to_csv('task_g_results.csv')