from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    #parser.add_argument('-t', '--times', default=4) # total number of images as multiple of original
    args = parser.parse_args()

    data_path = osp.join('dataset', args.cfg, 'data.yaml')
    cfg_path = osp.join('configs', args.cfg + '.yaml')

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg

    """ Change these lines per ablation in addition to the train config (and global config) """

    model = YOLO("yolo11s.yaml").load(cfg.model.weights)
    project_name = "run2"
    experiment_name = "train2"
    overwrite_run = False
    train_cfg = osp.join('training', project_name + '_' + experiment_name + '_cfg.yaml')

    """ --------------------------------------------------------------------------------- """

    results = model.train(data=data_path, cfg=train_cfg, project=project_name, name=experiment_name, exist_ok=overwrite_run)
