import torch
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from PIL import Image
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances


class CustomImageDataset:
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.classes = ["0", "1"]  # Define your class names here

        # You may need additional initialization here

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.image_dir)[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Define annotations (replace this with your annotation logic)
        mask_file = os.path.join(
            self.mask_dir, os.path.splitext(img_name)[0] + ".png"
        )
        masks = []
        with open(mask_file, "r") as f:
            for line in f:
                bbox_info = line.strip().split()
                class_name = bbox_info[0]
                x_min, y_min, x_max, y_max = map(float, bbox_info[1:])
                class_id = self.classes.index(class_name)
                masks.append(
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
            "masks": masks,
        }

        return record


# Register the dataset
def register_custom_dataset(name, image_dir, annotation_dir):
    DatasetCatalog.register(name, lambda: CustomImageDataset(image_dir, annotation_dir))
    MetadataCatalog.get(name).set(thing_classes=["0", "1"])  # Set your class names here


cfg = get_cfg()
cfg.merge_from_file(
    "/ghome/group04/MCV-C5-G4/week2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.025
cfg.SOLVER.MAX_ITER = 400
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

register_custom_dataset(
    "train_dataset",
    "/ghome/group04/MCV-C5-G4/week2/mask_rcnn/dataset/train/images",
    "/ghome/group04/MCV-C5-G4/week2/mask_rcnn/dataset/train/masks",
)
register_custom_dataset(
    "val_dataset",
    "/ghome/group04/MCV-C5-G4/week2/mask_rcnn/dataset/test/images",
    "/ghome/group04/MCV-C5-G4/week2/mask_rcnn/dataset/test/masks",
)

cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ("val_dataset",)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# evaluate
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

OUTPUT_EVAL_DIR = "./output/eval"
evaluator = COCOEvaluator("val_dataset", cfg, False, output_dir=OUTPUT_EVAL_DIR)
val_loader = build_detection_test_loader(cfg, "val_dataset")
inference_on_dataset(trainer.model, val_loader, evaluator)


# for d in dataset_dicts:  # dataset_dicts is the list of your custom dataset
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow("output", out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# cv2.destroyAllWindows()
