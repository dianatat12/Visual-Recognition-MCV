import json
import os
from detectron2.structures import BoxMode
from tqdm import tqdm

# #  Iimport json
# from pycocotools import mask as maskUtils
# import os
# import json
# from detectron2.structures import BoxMode


# def decompose_rle(rle_string, image_dims):
#     return {"counts": rle_string, "size": image_dims}


# def decompose_annotations(annotations):
#     frame_numbers = [int(annotation[0]) for annotation in annotations]
#     class_numbers = [int(annotation[1]) // 1000 for annotation in annotations]
#     image_dims = [
#         [int(annotation[3]), int(annotation[4])] for annotation in annotations
#     ]
#     rles = [annotation[-1].strip() for annotation in annotations]

#     return frame_numbers, class_numbers, image_dims, rles


# val_indexes = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]


# def create_coco_dataset(annotations_dir, images_dir, output_json_path):
#     coco_dataset = {
#         "info": {"description": "KITTI-MOTS Dataset in COCO format"},
#         "licenses": [],
#         "images": [],
#         "annotations": [],
#         "categories": [{"id": 1, "name": "pedestrian", "supercategory": "person"}],
#     }

#     annotation_id = 1  # Unique ID for each annotation
#     image_id = 1  # Unique ID for each image

#     # Process annotations and images
#     for file in sorted(os.listdir(annotations_dir)):
#         file_nr_str = file.split(".")[0]

#         split = "test" if file_nr_str in val_indexes else "train"
#         annotations_file = os.path.join(annotations_dir, file)
#         # images_path = os.path.join(images_dir, file_nr_str)

#         with open(annotations_file, "r") as fp:
#             annotations = fp.readlines()
#         annotations = [line.split(" ") for line in annotations]

#         frame_numbers, class_numbers, image_dims, rles = decompose_annotations(
#             annotations
#         )

#         for i, (frame_nr, class_nr) in enumerate(zip(frame_numbers, class_numbers)):
#             if int(class_nr) != 10:
#                 continue  # Skip non-pedestrian classes

#             frame_nr_str = "{:06d}".format(frame_nr)
#             rle = rles[i]
#             size = image_dims[i]
#             bbox = [int(coord) for coord in maskUtils.toBbox(decompose_rle(rle, size))]
#             area = bbox[2] * bbox[3]  # width * height

#             # Add image entry
#             image_entry = {
#                 "id": image_id,
#                 "file_name": f"{file_nr_str}_{frame_nr_str}.png",
#                 "width": size[1],
#                 "height": size[0],
#             }
#             coco_dataset["images"].append(image_entry)

#             # Add annotation entry
#             annotation_entry = {
#                 "id": annotation_id,
#                 "image_id": image_id,
#                 "category_id": 1,  # Assuming 1 is the ID for pedestrian
#                 # "segmentation": decompose_rle(rle, size),
#                 "area": area,
#                 "bbox": bbox,
#                 "iscrowd": 0,
#                 "bbox_mode": BoxMode.XYXY_ABS,
#             }
#             coco_dataset["annotations"].append(annotation_entry)

#             annotation_id += 1
#             image_id += 1

#     # Save COCO dataset to a JSON file
#     with open(output_json_path, "w") as json_file:
#         json.dump(coco_dataset, json_file, indent=4)

#     print(f"COCO dataset created at {output_json_path}")


# # Define your directories and output file path
# annotations_dir = "/ghome/group04/MCV-C5-G4/week2/dataset/KITTI-MOTS/instances_txt"
# images_dir = "/ghome/group04/MCV-C5-G4/week2/dataset/KITTI-MOTS/training/image_02"
# output_json_path = "coco_dataset.json"

# create_coco_dataset(annotations_dir, images_dir, output_json_path)

import cv2


def convert_to_coco(images_dir, labels_dir, output_json_path):
    # os.remove(output_json_path)
    coco_dataset = {
        "info": {"description": "Custom Dataset in COCO format"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "car", "supercategory": "object"},
            {"id": 1, "name": "pedestrian", "supercategory": "object"},
        ],
    }
    annotation_id = 1  # Unique ID for each annotation
    image_id = 1  # Unique ID for each image

    # val_indexes = [
    #     "0002",
    #     "0006",
    #     "0007",
    #     "0008",
    #     "0010",
    #     "0013",
    #     "0014",
    #     "0016",
    #     "0018",
    # ]

    # Iterate over the label files
    for label_file in tqdm(os.listdir(labels_dir)):
        if label_file.endswith(".txt"):
            # Get corresponding image file
            image_file = os.path.join(images_dir, label_file.replace(".txt", ".png"))
            if not os.path.exists(image_file):
                print(f"Image file not found for label file: {label_file}")
                continue

            # Read the image to get its size
            image = cv2.imread(image_file)
            if image is None:
                print(f"Failed to read image: {image_file}")
                continue
            height, width, _ = image.shape

            # Initialize a list to store annotations for the current image
            annotations = []

            # Open the label file
            with open(os.path.join(labels_dir, label_file), "r") as f:
                lines = f.readlines()

                # Iterate over the lines in the label file
                for line in lines:
                    # Split the line and extract the bounding box coordinates
                    class_id, x_center, y_center, bbox_width, bbox_height = map(
                        float, line.strip().split()
                    )
                    if int(class_id) not in [0, 1]:
                        continue

                    # Calculate bounding box coordinates for Detectron2
                    x1 = int((x_center - bbox_width / 2) * width)
                    y1 = int((y_center - bbox_height / 2) * height)
                    x2 = int((x_center + bbox_width / 2) * width)
                    y2 = int((y_center + bbox_height / 2) * height)

                    # Calculate area
                    area = bbox_width * bbox_height

                    # Add annotation entry
                    annotation_entry = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),  # Assuming 1 is the ID for object
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                        "area": area,
                        "iscrowd": 0,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                    annotations.append(annotation_entry)
                    annotation_id += 1

            # Add image entry
            image_entry = {
                "id": image_id,
                "file_name": os.path.basename(image_file),
                "width": width,
                "height": height,
            }
            coco_dataset["images"].append(image_entry)
            coco_dataset["annotations"].extend(annotations)
            image_id += 1

    # Save COCO dataset to a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(coco_dataset, json_file, indent=4)

    print(f"COCO dataset created at {output_json_path}")


images_dir = "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/train/images"
labels_dir = "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/train/labels"
output_json_path = "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/train/coco.json"
convert_to_coco(images_dir, labels_dir, output_json_path)


images_dir = "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/test/images"
labels_dir = "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/test/labels"
output_json_path = "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/test/coco.json"
convert_to_coco(images_dir, labels_dir, output_json_path)
