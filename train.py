from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='any_v2')
    #parser.add_argument('-t', '--times', default=4) # total number of images as multiple of original
    args = parser.parse_args()

    data_path = osp.join('dataset', args.cfg, 'data.yaml')
    cfg_path = osp.join('configs', args.cfg + '.yaml')

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg
    #train_from_weights = "first_run/train/weights/best.pt"
    model = YOLO("training/yolo11m.pt")

    overwrite_run = False
    train_cfg = osp.join('training', 'cfg.yaml')

    results = model.train(data=data_path, cfg=train_cfg, exist_ok=overwrite_run)
