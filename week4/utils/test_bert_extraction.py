from transformers import BertTokenizer, BertModel
import torch
from torchvision import transforms, models
import datetime
import numpy as np
from PIL import Image
from utils import match_items, read_captions_json, create_captions_dataframe


def get_bert_embeddings(text, tokenizer, model):

    # Tokenize input text
    tokens = tokenizer(text, return_tensors="pt")

    # Get BERT model embeddings
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the embeddings for the first token ([CLS] token)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = embeddings.numpy()

    return embeddings


def get_resnet_features(img, model, transform):
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        features = model(img)
    features = features.squeeze(0).squeeze(1).squeeze(1).numpy()
    return features


text_embeddings_destination = "/home/georg/projects/university/C5_Visual_Recognition/MCV-C5-G4/dataset/bert_large_embeddings/"
image_embeddings_destination = "/home/georg/projects/university/C5_Visual_Recognition/MCV-C5-G4/dataset/resnet50_embeddings/"
for split in ["val", "train"]:
    # Load captions data
    image_data, caption_data = read_captions_json(
        f"/home/georg/projects/university/C5_Visual_Recognition/MCV-C5-G4/dataset/COCO/captions_{split}2014.json"
    )
    data_list = match_items(image_data, caption_data)
    df = create_captions_dataframe(data_list)

    image_directory = f"/home/georg/projects/university/C5_Visual_Recognition/MCV-C5-G4/dataset/COCO/{split}2014/"

    version = "bert-large-uncased"
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(version)
    model = BertModel.from_pretrained(version)

    resnet50 = models.resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
    print(resnet50)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # iterate over rows in val_df
    for index, row in df.iterrows():
        # Get BERT embeddings for the caption
        caption = row["caption"]
        text_embeddings = get_bert_embeddings(caption, tokenizer=tokenizer, model=model)
        image_id = row["image_id"]
        # Save the embeddings to a file
        filename = f"{text_embeddings_destination}{split}/{image_id}.npy"
        with open(filename, "wb") as f:
            np.save(f, text_embeddings)

        img_file_name = row["file_name"]
        img_path = image_directory + img_file_name
        img = Image.open(img_path).convert("RGB")
        features = get_resnet_features(img, resnet50, transform)
        print(features.shape)
        # Save the embeddings to a file
        filename = f"{image_embeddings_destination}{split}/{image_id}.npy"
        with open(filename, "wb") as f:
            np.save(f, features)
