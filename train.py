from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    parser.add_argument('-t', '--times', default=4) # total number of images as multiple of original
    args = parser.parse_args()

    data_path = osp.join('dataset', args.cfg, 'data.yaml')
    cfg_path = osp.join('configs', args.cfg + '.yaml')

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg

    model = YOLO("yolo11s.yaml").load("yolo11s.pt")

    results = model.train(data=data_path, cfg="default.yaml")
  
    """ results = model.train(data=data_path,
                          epochs=cfg.train.epochs,
                          batch=cfg.train.batch_size,
                          imgsz=cfg.model.input_size,
                          project='first_run',
                          device='mps',
                          optimizer=cfg.train.optimizer,
                          seed=cfg.train.seed,
                          mosaic=cfg.train.close_mosaic,
                          cos_lr=cfg.train.cos_lr,
                          plots=cfg.train.plots,
                          ) """
    
      
    """results = model.train(data=cfg,
                          augment=False,  # Disable augmentations
                          hsv_h=0,  # Disable HSV color augmentations
                          hsv_s=0,
                          hsv_v=0,
                          degrees=0,  # Disable rotation
                          translate=0,  # Disable translation
                          scale=0,  # Disable scaling
                          shear=0,  # Disable shearing
                          perspective=0,  # Disable perspective transform
                          flipud=0,  # Disable vertical flip
                          fliplr=0,  # Disable horizontal flip
                          mosaic=0,  # Disable mosaic augmentation
                          mixup=0,  # Disable mixup augmentation
                          copy_paste=0  # Disable copy-paste augmentation
                      ) """
    
    """results = model.train(
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

    from claude:
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
    p: 0.5
    
    or like this from github forum: https://github.com/ultralytics/ultralytics/issues/14042
    train:
  augmentations:
    - type: albumentations
      transforms:
        - RandomCrop: {height: 512, width: 512}
        - HorizontalFlip: {p: 0.5}
        - RandomBrightnessContrast: {p: 0.2}"""
