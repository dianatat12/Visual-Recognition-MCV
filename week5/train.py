from utils.model import ImageTextRetrievalModel
from utils.data import ImageTextDataModule
from pytorch_lightning import Trainer

from utils.transforms import albumentations_transform

text2imgmodel = ImageTextRetrievalModel(
    text_embedding_dim=768,
    image_embedding_dim=2048,
    learning_rate=1e-3,
    mode="text2img",
)

img2textmodel = ImageTextRetrievalModel(
    text_embedding_dim=768,
    image_embedding_dim=2048,
    learning_rate=1e-4,
    mode="img2text",
)

dm = ImageTextDataModule(
    batch_size=64, num_workers=4, transform=None  # lbumentations_transform()
)


trainer = Trainer(max_epochs=1, accelerator="auto", log_every_n_steps=10)

trainer.fit(text2imgmodel, dm)

trainer = Trainer(max_epochs=1, accelerator="auto", log_every_n_steps=10)
trainer.fit(img2textmodel, dm)
