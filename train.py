from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp

# maybe not even needed
def albumentations_transform(image, *args, **kwargs):
    deepdartsAugmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(shear={"x": (-30, 30), "y": (-30, 30)}, p=0.5),  # Warping with shear
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Jitter
    ])
    transformed = deepdartsAugmentations(image=image)
    return transformed['image']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    args = parser.parse_args()

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    model = YOLO("yolo11m-pose.yaml").load("yolo11m-pose.pt")

    

    transform = A.compose(deepdartsAugmentations, p=cfg.aug.overall_prob)

    results = model.train(data=cfg,
                        epochs=cfg.train.epochs,
                        batch=cfg.train.batch_size,
                        imgsz=cfg.model.input_size,
                        device='mps',
                        project=cfg.train.name,
                        optimizer=cfg.train.optimizer,
                        seed=cfg.train.seed,
                        close_mosaic=cfg.train.mosaic,
                        cos_lr=cfg.train.cos_lr,
                        plots=cfg.train.plots,
                        augment=False,
                        transforms=deepdartsAugmentations
                        )
    
    """ results = model.train(
    data=cfg,
    # ... other parameters ...
    augment=True,  # Enables default augmentations
    # You can customize specific augmentation parameters
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.7,    # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,    # image HSV-Value augmentation (fraction)
    degrees=0.0,  # image rotation (+/- deg)
    translate=0.1,# image translation (+/- fraction)
    scale=0.5,    # image scale (+/- gain)
    shear=0.0,    # image shear (+/- deg)
    perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
    flipud=0.0,   # image flip up-down (probability)
    fliplr=0.5,   # image flip left-right (probability)
    mosaic=1.0,   # image mosaic (probability)
    mixup=0.0,    # image mixup (probability)
    ) 
    albumentations:
  horizontal_flip: 0.5
  vertical_flip: 0.0
  rotate90: 0.5
  affine:
    shear:
      x: [-30, 30]
      y: [-30, 30]
    p: 0.5
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p: 0.5"""
