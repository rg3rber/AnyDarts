from ultralytics import YOLO
import albumentations as A
import cv2
import argparse
from yacs.config import CfgNode
import os.path as osp
from HelperFunctions import bboxes_to_xy, get_dart_scores
import numpy as np
from BoardDetection import find_board
from inference import inference
from time import time

""" Pipeline: 1. find board, 2. crop to board, 3. run inference and get scores, 4. return scores """

def crop(img, boardbbox, cfg):
    """
    img: image to crop
    board_bbox: bounding box of the board
    cfg: config file
    returns: cropped image
    """
    size = cfg.model.input_size
    crop = img[boardbbox[0]:boardbbox[1], boardbbox[2]:boardbbox[3]]
    crop = cv2.resize(crop, (size, size))
    return crop

def score(img, cfgFile="holodarts", debug=False, img_name=None):
    """
    img: original image or path to image to run inference on
    returns: scores of the darts in the image
    """
    if isinstance(img, str) and osp.isfile(img): # img is path to image else img is already loaded
        img = cv2.imread(img)
    
    if img is None:
        print('Could not open or find the image:', img)
        return -1
    
    if debug:
        cv2.imshow("score debug: original", img)
        cv2.waitKey()

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', cfgFile + '.yaml'))

    #TODO: once model is working and path is inside config: model = YOLO(cfg.model.name)
    #old: model = YOLO('models/holo_train2/weights/best.pt')
    model = YOLO("first_run/mosaic_v22/weights/best.pt")

    boardbbox = find_board(img)
    if boardbbox is None:
        return -1
    
    board = crop(img, boardbbox, cfg)
    if debug: 
        cv2.imshow("score debug board:", board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    starttime = time()
    score = inference(model, board, cfg, img_name=img_name)
    print(f"Inference time taken: {time() - starttime}")

    return score

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Board detection using ellipse fitting.')
    parser.add_argument('-i', '--input', help='Image name inside dataset/images', 
                       default='dataset/images/new-test.jpg')
    parser.add_argument('-c', '--cfg', help='Config file name', default='holo_v1')
    args = parser.parse_args()

    img = args.input
    imgPrompt = input("Enter the image name or path to the image or leave blank to use: " + img)
    if osp.isfile(imgPrompt):
        img = imgPrompt
        print("using imgprompt path: ", img)
    if osp.isfile(osp.join('dataset/images', imgPrompt)):
        img = osp.join('dataset/images', imgPrompt)
        print("using imgprompt name: ", img)

    preloaded_img = cv2.imread(img)
    if preloaded_img is None:
        print("Invalid image path: ", img)
        exit(1)

    print("Scoring image as path: " , img, " with config: " + args.cfg)
    score_imgAsPath = score(img, args.cfg, True)
    print("Scoring preloaded image: " , img, " with config: " + args.cfg)
    #score_imgLoaded = score(preloaded_img, args.cfg, True)

    print(f"Score from image as path: {score_imgAsPath}")
    #print(f"Score from image loaded: {score_imgLoaded}")