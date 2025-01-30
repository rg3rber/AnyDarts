from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp
from HelperFunctions import bboxes_to_xy, get_dart_scores
import numpy as np


def inference(model, img_board, cfg):
    """
    img_board: original image cropped to board
    """
    if isinstance(img_board, str) and osp.isfile(img_board):
        img = cv2.imread(img_board)
        cv2.imshow("Inference using path before: ", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Inference using path after: ", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imshow("Inference using loaded image before: ", img_board)
        img = cv2.cvtColor(img_board, cv2.COLOR_BGR2RGB)
        cv2.imshow("Inference using loaded image after: ", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    results = model(img, max_det=4+cfg.model.max_darts)
    results[0].show()
    
    boxes = results[0].boxes

    xywh = boxes.xywh.numpy()
    classes = boxes.cls.numpy()
    bboxes = np.concatenate((xywh, classes[:, None]), axis=1)
    xy, _ = bboxes_to_xy(bboxes, cfg.model.max_darts)
    scores = get_dart_scores(xy, cfg, True)
    score = np.sum(scores)

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    parser.add_argument('-m', '--model', default='first_run/train/weights/best.pt')
    parser.add_argument('-i', '--image', default='dataset/holo_v1/test/images/d1_03_27_2020_IMG_6135.JPG')
    args = parser.parse_args()

    model = YOLO(args.model)

    cfg_path = osp.join('configs', args.cfg + '.yaml')

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg

    image_path = args.image
    predicted_score = inference(model, image_path, cfg)
    print(predicted_score)