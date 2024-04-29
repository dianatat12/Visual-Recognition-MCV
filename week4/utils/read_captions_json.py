import os
import json
import pandas as pd


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


dataset_path = "/ghome/group04/mcv/datasets/C5/COCO/"
train_captions_file = os.path.join(dataset_path, "captions_train2014.json")
val_captions_file = os.path.join(dataset_path, "captions_val2014.json")

train_img_data, train_caption_data = read_captions_json(train_captions_file)
val_img_data, val_caption_data = read_captions_json(val_captions_file)

train_data = match_items(train_img_data, train_caption_data)
val_data = match_items(val_img_data, val_caption_data)

train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)

print(train_df.head())
print(val_df.head())

train_df = train_df.drop(
    columns=["license", "coco_url", "flickr_url", "date_captured", "height", "width"]
)
val_df = val_df.drop(
    columns=["license", "coco_url", "flickr_url", "date_captured", "height", "width"]
)


print(train_df.columns)
print(train_df.head())
print(val_df.head())
