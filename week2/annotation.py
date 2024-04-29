import os
import cv2
from tqdm import tqdm

# Define the directory paths


# Initialize an empty dictionary to store image and bounding box information
def get_annotation(
    images_dir="/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/test/images",
    labels_dir="/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/test/labels",
):
    data_dict = {}

    # Iterate over the label files
    for label_file in tqdm(os.listdir(labels_dir)[:3]):
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

            # Initialize a list to store bounding boxes for the current image
            bbox_list = []

            # Open the label file
            with open(os.path.join(labels_dir, label_file), "r") as f:
                lines = f.readlines()

                # Iterate over the lines in the label file
                for line in lines:
                    # Split the line and extract the bounding box coordinates
                    class_id, x_center, y_center, bbox_width, bbox_height = map(
                        float, line.strip().split()
                    )

                    # Calculate bounding box coordinates for Detectron2
                    x1 = int((x_center - bbox_width / 2) * width)
                    y1 = int((y_center - bbox_height / 2) * height)
                    x2 = int((x_center + bbox_width / 2) * width)
                    y2 = int((y_center + bbox_height / 2) * height)

                    # Add the bounding box coordinates to the list
                    bbox_list.append([x1, y1, x2, y2, int(class_id)])

            # Add the image and bounding box information to the dictionary
            data_dict[image_file] = bbox_list
    return data_dict


# create a function which convert the above function to convert data into detectron2 coco format
def convert_to_coco_format(data_dict):
    # Initialize a list to store the annotations
    annotations = []

    # Initialize a dictionary to store the categories
    categories = {}

    # Initialize a counter for the annotation IDs
    annotation_id = 0

    # Iterate over the images in the data dictionary
    for image_file, bbox_list in data_dict.items():
        # Get the image ID
        image_id = int(image_file.split("/")[-1].split(".")[0])

        # Iterate over the bounding boxes
        for bbox in bbox_list:
            # Extract the bounding box coordinates and class ID
            x1, y1, x2, y2, class_id = bbox

            # Create the annotation in COCO format
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }

            # Add the annotation to the list
            annotations.append(annotation)

            # Increment the annotation ID
            annotation_id += 1

            # Add the category to the dictionary
            categories[class_id] = {"id": class_id, "name": str(int(class_id))}

    # Create the COCO format dictionary
    coco_format = {
        "images": [{"id": int(image_file.split("/")[-1].split(".")[0])}],
        "annotations": annotations,
        "categories": list(categories.values()),
    }

    return coco_format
