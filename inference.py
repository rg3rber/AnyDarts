from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp
from HelperFunctions import bboxes_to_xy, get_dart_scores, draw
import numpy as np


def inference(model, img_board, cfg, img_name=None):
    """
    img_board: original image cropped to board
    """
    if isinstance(img_board, str) and osp.isfile(img_board):
        img = cv2.imread(img_board)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for reasons unclear up to now prediction works with image directly loaded from cv2
    else:
        #img = cv2.cvtColor(img_board, cv2.COLOR_BGR2RGB)
        img = img_board

    results = model(img, max_det=4+cfg.model.max_darts)

    boxes = results[0].boxes
    xywh = boxes.xywh.numpy()
    classes = boxes.cls.numpy()
    bboxes = np.concatenate((xywh, classes[:, None]), axis=1)
    xy, _ = bboxes_to_xy(bboxes, cfg.model.max_darts)
    
    scores = get_dart_scores(xy, cfg, True)
    score = np.sum(scores)

    if img_name is not None:
        img_base_name = osp.splitext(osp.basename(img_name))[0]
        img_dir = osp.dirname(img_name)
        predicted_filename = osp.join(img_dir, f"{img_base_name}_pred.jpg")
        img_with_pred = draw(img_board.copy(), xy[:, :2], cfg, circles=False, score=score)
        cv2.imwrite(predicted_filename, img_with_pred)

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='any_v1')
    parser.add_argument('-m', '--model', default='first_run/train/weights/best.pt')
    parser.add_argument('-i', '--image', default='dataset/any_v1/test/images/d1_03_27_2020_IMG_6135.JPG')
    args = parser.parse_args()

    model = YOLO(args.model)

    cfg_path = osp.join('configs', args.cfg + '.yaml')

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg

    image_path = args.image
    predicted_score = inference(model, image_path, cfg)
    print(predicted_score)