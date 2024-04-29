import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
from tqdm import tqdm

from annotation import get_annotation
from sklearn.metrics import average_precision_score


def get_prediction(image_path):
    image = cv2.imread(image_path)
    outputs = predictors(image)
    return outputs

def calculate_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)

    iou = intersection / float(pred_area + gt_area - intersection)
    return iou