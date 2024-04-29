import os
import json
import pandas as pd
import torch


def match_items(list1, list2):
    # Create a dictionary from list2 with 'image_id' as the key
    dict2 = {item["image_id"]: item for item in list2}

    # Iterate over list1 and add the corresponding item from list2
    for item in list1:
        if item["id"] in dict2:
            item.update(dict2[item["id"]])

    return list1


def read_captions_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    image_data = data["images"]
    caption_data = data["annotations"]
    return image_data, caption_data


def create_captions_dataframe(list):
    df = pd.DataFrame(list)
    df = df.drop(
        columns=[
            "license",
            "coco_url",
            "flickr_url",
            "date_captured",
            "height",
            "width",
        ]
    )
    return df


def get_image_encoder_output_size(image_encoder):
    random_input = torch.randn(1, 3, 224, 224)
    resnet_output = image_encoder(random_input)

    return resnet_output.shape[0]


def get_text_encoder_output_size(text_encoder):
    input_text = "This is a sample sentence."
    bert_output = text_encoder(input_text)

    return bert_output.shape[0]
