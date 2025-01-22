from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    args = parser.parse_args()

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    model = YOLO("yolo11m-pose.yaml").load("yolo11m-pose.pt")

    deepdartsAugmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(shear={"x": (-30, 30), "y": (-30, 30)}, p=0.5),  # Warping with shear
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Jitter
    ToTensorV2(),  # Convert to tensor
])

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
                        augmentations=deepdartsAugmentations
                        )
