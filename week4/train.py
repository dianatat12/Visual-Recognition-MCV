from model import ImageTextRetrievalModel
from data import ImageTextDataModule
from pytorch_lightning import Trainer

from transforms import albumentations_transform

text2imgmodel = ImageTextRetrievalModel(
    text_embedding_dim=768,
    image_embedding_dim=2048,
    learning_rate=1e-4,
    mode="text2img",
)

img2textmodel = ImageTextRetrievalModel(
    text_embedding_dim=768,
    image_embedding_dim=2048,
    learning_rate=1e-4,
    mode="img2text",
)

dm = ImageTextDataModule(
    batch_size=32, num_workers=4, transform=albumentations_transform()
)


trainer = Trainer(
    max_epochs=1, accelerator="auto", log_every_n_steps=10, val_check_interval=0.25
)

trainer.fit(text2imgmodel, dm)

trainer = Trainer(
    max_epochs=1, accelerator="auto", log_every_n_steps=10, val_check_interval=0.25
)
trainer.fit(img2textmodel, dm)
