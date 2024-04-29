import os
import json
import pandas as pd
import random
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers import AutoPipelineForText2Image
import torch


def match_items(list1, list2):
    # Create a dictionary from list2 with 'image_id' as the key
    dict1 = {item["id"]: item for item in list1}
    # Iterate over list1 and add the corresponding item from list2
    for item in list2:
        if item["image_id"] in dict1:
            item.update(dict1[item["image_id"]])

    return list2


def read_captions_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    image_data = data["images"]
    caption_data = data["annotations"]
    return image_data, caption_data


def prepare_data(json):
    image_data, caption_data = read_captions_json(json)
    print(f"Image data rows: {len(image_data)}")
    print(f"Caption data rows: {len(caption_data)}")
    data_list = match_items(image_data, caption_data)
    df = pd.DataFrame(data_list)
    return df


json_path = "/ghome/group04/mcv/datasets/C5/COCO/captions_train2014.json"
df = prepare_data(json=json_path)

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")
pipe = pipe.to("cuda")
print(df.head())

coco_animals = [
    " bird ",
    " cat ",
    " dog ",
    " horse ",
    " sheep ",
    " cow ",
    " elephant ",
    " zebra ",
    " giraffe ",
    " birds ",
    " cats ",
    " dogs ",
    " horses ",
    " sheep ",
    " cows ",
    " elephants ",
    " zebras ",
    " giraffes ",
]

new_animals = [
    " deer ",
    " tortoise ",
    " camel ",
    " fox ",
    " polar bear ",
    " panda ",
    " parrot ",
    " goat ",
    " kangaroo ",
    " moose ",
]

animal_counts = {
    animal: 0 for animal in coco_animals
}  # Initialize count for each animal
i = 0
for index, row in df.iterrows():
    caption = row[
        "caption"
    ].lower()  # Convert caption to lowercase for case-insensitive matching
    # Check if any animal is mentioned in the caption
    for animal in coco_animals:
        if animal in caption:
            animal_counts[animal] += 1  # Increment count for the found animal
            print("++++++++++++++++++++")
            print(caption)
            prompt = f"Photograph of " + caption.replace(
                animal, random.choice(new_animals)
            )
            print(prompt)
            image = pipe(
                prompt=prompt, num_inference_steps=3, guidance_scale=0.0
            ).images[0]

            # image = pipe(prompt).images[0]
            image.save(f"sdxl-turbo-images/{caption.replace(' ', '-')}.png")
            i += 1

            print("++++++++++++++++++++\n")

# Print the counts for each animal
for animal, count in animal_counts.items():
    print(f"{animal.capitalize()}: {count} occurrences")
