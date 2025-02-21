import albumentations as A
import cv2
import numpy as np
import argparse
from yacs.config import CfgNode
import os.path as osp

def augmentations(image_path, label_path, cfg):
    # Separate bboxes and labels by class
    img = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    classes = cfg.names
    class_labels = [] # [cal1, cal2, cal3, cal4, (dart), (dart), (dart)]
    bboxes = [] # bboxes[:4] = calibration points, bboxes[4:] = darts
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        bboxes.append([float(x) for x in line[1:]]) # x_center, y_center, width, height
        class_labels.append(classes[i])

    calib_points = bboxes[:4]
    darts = bboxes[4:]


    """insert deepdarts augmentations here"""

    """add additional augmentations here"""
    
    # Apply transformations only to darts and image
    augmented = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
    
    augmented_result = augmented(image=img, bboxes=darts, class_labels=class_labels)
    
    # Combine fixed calibration points with augmented darts
    final_bboxes = calib_points + augmented_result['bboxes']
    final_labels = [label for label in labels if label in [1,2,3,4]] + [5]
    
    return augmented_result['image'], final_bboxes, final_labels

    deepdartsAugmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(shear={"x": (-30, 30), "y": (-30, 30)}, p=0.5),  # Warping with shear
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Jitter
    ])
    transformed = deepdartsAugmentations(image=image)
    return transformed['image']

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    parser.add_argument('-t', '--times', default=4) # total number of images as multiple of original
    args = parser.parse_args()

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(osp.join('../configs', f'{args.cfg}.yaml'))
    cfg.model.name = args.cfg




