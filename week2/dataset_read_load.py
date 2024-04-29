import torch
import detectron2

import pycocotools.mask as rletools
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import shutil

#get KITTI-MOTS data to the required Detectron2 input format
images_dir = '/ghome/group04/MCV-C5-G4/week2/dataset/KITTI-MOTS/training/image_02'
annotations_dir = '/ghome/group04/MCV-C5-G4/week2/dataset/KITTI-MOTS/instances_txt'
masks_dir = '/ghome/group04/MCV-C5-G4/week2/dataset/KITTI-MOTS/instances'

val_indexes = [ '0002',
                '0006', 
                '0007', 
                '0008', 
                '0010', 
                '0013', 
                '0014', 
                '0016', 
                '0018'
                ]


def decompose_rle(rle_string, image_dims):    
    return {'counts':rle_string, 'size':image_dims}

def decompose_annotations(annotations):
    frame_numbers = [int(annotation[0]) for annotation in annotations]
    class_numbers = [int(annotation[1]) // 1000 for annotation in annotations]
    image_dims = [[int(annotation[3]), int(annotation[4])] for annotation in annotations]
    rles = [annotation[-1].strip() for annotation in annotations]

    return frame_numbers, class_numbers, image_dims, rles

def xyhw2xyxy(coordinates):    
    x,y,w,h = coordinates

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def xyhw2detectron(coordinates, dims):    
    x,y,w,h = coordinates

    x = (x + w/2) / dims[1]
    y = (y + h/2) / dims[0]
    w = w / dims[1]
    h = h / dims[0]
    return [x, y, w, h]



output_directory = './dataset'
os.makedirs(output_directory)
os.makedirs(output_directory + '/train')
os.makedirs(output_directory + '/test')
os.makedirs(output_directory + '/train/images')
os.makedirs(output_directory + '/train/labels')
os.makedirs(output_directory + '/test/images')
os.makedirs(output_directory + '/test/labels')



for file in sorted(os.listdir(annotations_dir)):
    file_nr_str = file.split('.')[0]

    split = 'test' if file_nr_str in val_indexes else 'train'
    destination = os.path.join(output_directory, split)

    annotations_file = os.path.join(annotations_dir, file)
    images_path = os.path.join(images_dir, file_nr_str) 

    with open(annotations_file, 'r') as fp:
        annotations = fp.readlines()
    annotations = [line.split(' ') for line in annotations]
    
    frame_numbers, class_numbers, image_dims, rles = decompose_annotations(annotations)
    for i, (frame_nr, class_nr) in enumerate(zip(frame_numbers, class_numbers)):

        if int(class_nr) != 10:
            frame_nr = '{:06d}'.format(frame_nr)
            rle = rles[i]
            size = image_dims[i]
            bbox = [int(coord) for coord in rletools.toBbox(decompose_rle(rle, size))]  
            detectron_bbox = xyhw2xyxy(bbox)

            class_nr_adjusted = str(class_nr - 1)
            
            ann_string = f'{class_nr_adjusted} {detectron_bbox[0]} {detectron_bbox[1]} {detectron_bbox[2]} {detectron_bbox[3]}\n'
            file_name = os.path.join(destination, 'labels', f'{file_nr_str}_{frame_nr}.txt')

            with open(file_name, 'a+')as fp:
                fp.write(ann_string)

            img_src = os.path.join(images_path, f'{frame_nr}.png')
            img_dest = os.path.join(destination, 'images', f'{file_nr_str}_{frame_nr}.png')
            shutil.copy(img_src, img_dest)








