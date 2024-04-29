import os
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image
from detectron2.structures import BoxMode


class CustomImageDataset:
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = ["0", "1"]  # Define your class names here

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.image_dir)[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        print(f"Loading image: {img_path}")
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Define annotations (replace this with your annotation logic)
        annotation_file = os.path.join(
            self.annotation_dir, os.path.splitext(img_name)[0] + ".txt"
        )
        annotations = []
        with open(annotation_file, "r") as f:
            for line in f:
                bbox_info = line.strip().split()
                class_name = bbox_info[0]
                x_min, y_min, x_max, y_max = map(float, bbox_info[1:])
                class_id = self.classes.index(class_name)
                annotations.append(
                    {
                        "bbox": [x_min, y_min, x_max, y_max],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": class_id,
                    }
                )

        record = {
            "file_name": img_path,
            "height": img_height,
            "width": img_width,
            "image_id": idx,
            "annotations": annotations,
        }

        return record


# Register the dataset
def register_custom_dataset(name, image_dir, annotation_dir):
    print(f"Registering dataset: {name}")
    DatasetCatalog.register(name, lambda: CustomImageDataset(image_dir, annotation_dir))
    MetadataCatalog.get(name).set(thing_classes=["0", "1"])  # Set your class names here


register_custom_dataset(
    "test_dataset",
    "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/test/images",
    "/ghome/group04/MCV-C5-G4/week2/faster_rcnn/dataset/test/labels",
)

cfg = get_cfg()
cfg.merge_from_file(
    "/ghome/group04/MCV-C5-G4/week2/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.DATALOADER.NUM_WORKERS = 2

# Create a predictor for inference
print("Creating predictor...")
predictor = DefaultPredictor(cfg)

# Evaluate on the test dataset
OUTPUT_EVAL_DIR = "./output/eval"
print("Performing evaluation...")
evaluator = COCOEvaluator("test_dataset", cfg, False, output_dir=OUTPUT_EVAL_DIR)
val_loader = build_detection_test_loader(cfg, "test_dataset")
print("Running inference on the test dataset...")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
