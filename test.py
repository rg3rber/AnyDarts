from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='any_v1')
    args = parser.parse_args()

    data_path = osp.join('dataset', args.cfg, 'data.yaml')
    cfg_path = osp.join('configs', args.cfg + '.yaml')
    test_cfg_path = osp.join('testing', 'cfg.yaml')

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg

    test_cfg = CfgNode(new_allowed=True)
    test_cfg.merge_from_file(test_cfg_path)

    weights = test_cfg.model
    print(weights)
    
    model = YOLO("yolo11s.yaml").load(weights)

    overwrite_run = False

    results = model.val(data=data_path, cfg=test_cfg, exist_ok=overwrite_run)