import os
import json
import pandas as pd
import numpy as np


def match_items(list1, list2):
    # Create a dictionary from list2 with 'image_id' as the key
    dict2 = {item["image_id"]: item for item in list2}

    # Iterate over list1 and add the corresponding item from list2
    for item in list1:
        if item["id"] in dict2:
            item.update(dict2[item["id"]])

    return list1


def read_captions_json(json_path):
    classes = []
    with open(json_path, "r") as f:
        data = json.load(f)
    image_data = data["images"]
    caption_data = data["annotations"]

    for caption in caption_data:
        classes.append(caption["category_id"])
    print(sorted(list(set(classes))))
    return image_data, caption_data


dataset_path = (
    "/home/georg/projects/university/C5_Visual_Recognition/MCV-C5-G4/dataset/COCO"
)
train_captions_file = os.path.join(dataset_path, "instances_train2014.json")
val_captions_file = os.path.join(dataset_path, "instances_val2014.json")

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

dest = "/home/georg/projects/university/C5_Visual_Recognition/MCV-C5-G4/dataset/class_annotations/val"
nans = 0
rows = 0
# save annotations to .npy files for each image based in image_id
for index, row in val_df.iterrows():
    rows += 1
    # Get BERT embeddings for the caption
    try:
        classes = np.array([int(row["category_id"])])
    except:
        classes = np.array([-1])
        nans += 1
    image_id = row["image_id"]
    # Save the embeddings to a file
    filename = f"{dest}/{image_id}.npy"
    with open(filename, "wb") as f:
        np.save(f, classes)

print(nans)
print(rows)


dest = "/home/georg/projects/university/C5_Visual_Recognition/MCV-C5-G4/dataset/class_annotations/train"
nans = 0
rows = 0
# save annotations to .npy files for each image based in image_id
for index, row in train_df.iterrows():
    rows += 1
    # Get BERT embeddings for the caption
    try:
        classes = np.array([int(row["category_id"])])
    except:
        classes = np.array([-1])
        nans += 1
    image_id = row["image_id"]
    # Save the embeddings to a file
    filename = f"{dest}/{image_id}.npy"
    with open(filename, "wb") as f:
        np.save(f, classes)

print(nans)
print(rows)
