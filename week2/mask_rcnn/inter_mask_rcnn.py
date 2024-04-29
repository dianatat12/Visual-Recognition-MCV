import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# TASK C. INFERENCE 

# Function to get configuration
def get_mask_rcnn_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("/ghome/group04/MCV-C5-G4/week2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Update with the correct path

    # Set model weights to COCO pretrained weights
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    # Set threshold for instance predictions
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Set device to CPU
    cfg.MODEL.DEVICE = 'cpu'

    return cfg

# Register the single image dataset to Detectron2
def load_single_image_dataset(image_path):
    image_id = "000004"  # You may want to extract this from the image path or use a different ID scheme
    dataset = [{"file_name": image_path, "image_id": image_id}]
    return dataset

DatasetCatalog.register("single_image_dataset", lambda: load_single_image_dataset())
MetadataCatalog.get("single_image_dataset").set(thing_classes=["car", "pedestrian"])

# Create a Mask R-CNN predictor using the modified configuration
mask_rcnn_cfg = get_mask_rcnn_cfg()
predictor = DefaultPredictor(mask_rcnn_cfg)

# Perform inference on the single example image
image_path = "/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/test/images/0002_000180.png"
image = cv2.imread(image_path)
outputs = predictor(image)

# Create a visualizer instance
visualizer = Visualizer(image, MetadataCatalog.get(mask_rcnn_cfg.DATASETS.TRAIN[0]), scale=1.2)

# Visualize the predictions on the image
vis_output = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
vis_image = vis_output.get_image()

# Save the visualized image
cv2.imwrite("inferred_image_with_masks.jpg", vis_image)
