import os.path as osp
import os
from ultralytics import YOLO
import cv2
import argparse
from yacs.config import CfgNode
from inference import inference
from HelperFunctions import bboxes_to_xy, get_dart_scores, draw
import numpy as np
from time import time
import pandas as pd

""" run inference on set of images"""

def batch_inference(model, path, cfg, test=False, write=False, fail_cases=False):
    if osp.isdir(path):
        images = os.listdir(path)
        img_paths = [osp.join(path, name) for name in images]
    else:
        test = False
        img_paths = [path]

    if test:
        write_dir = osp.join('test/predictions', cfg.model.name)
    else:
        write_dir = osp.join('custom_predictions', cfg.model.name)
    
    print(f'Making predictions with {cfg.model.name}...')
    
    # Timing for FPS calculation
    start_time = time()
    preds = np.zeros((len(img_paths), 4 + cfg.model.max_darts, 3))
    gt_xys = np.zeros((len(img_paths), 7, 3))
    estimated = np.array([-1] * len(img_paths)).tolist()

    ASE = []  # Absolute Score Error
    
    for i, p in enumerate(img_paths):
        img = cv2.imread(p)

        img_name, img_ext = osp.splitext(osp.basename(p))[0], osp.splitext(osp.basename(p))[1]
        
        results = model(p, max_det=4+cfg.model.max_darts) # YOLO expects RGB images so give it the path
        boxes = results[0].boxes
        xywh = boxes.xywh.numpy()
        classes = boxes.cls.numpy()
        bboxes = np.concatenate((xywh, classes[:, None]), axis=1)

        # Convert bboxes to xy format
        pred, hadToEstimate = bboxes_to_xy(bboxes, cfg.model.max_darts)
        if len(hadToEstimate) > 0:
            estimated[i] = hadToEstimate # list of calibration points that had to be estimated

        preds[i] = pred
        pred = pred[pred[:, -1]== 1] # remove invisible bboxes

        pred_score = sum(get_dart_scores(pred, cfg, numeric=True))
        
        # Handle ground truth if test mode
        if test:
            label_path = osp.join(osp.dirname(path), 'labels', 
                            osp.splitext(osp.basename(p))[0] + '.txt')
            if osp.exists(label_path):
                with open(label_path, 'r') as f: # get label: [class, x, y, w, h]
                    labels = np.array([list(map(float, line.strip().split())) 
                                     for line in f.readlines()])
                if len(labels):
                    # Convert YOLO format to xy coordinates
                    
                    gt_xy = np.concatenate((cfg.model.input_size * labels[:, 1:3], np.ones((labels.shape[0], 1))), axis=1)  # x, y, visible
                    pad_gt_xy = gt_xy.copy()
                    if len(labels) < 7:
                        pad_gt_xy = np.concatenate((gt_xy, np.zeros((7 - len(labels), 3)))) # pad with zeros
                    gt_xys[i] = pad_gt_xy.astype(np.float32)

                    # Calculate score error
                    gt_score = sum(get_dart_scores(gt_xy, cfg, numeric=True))
                    ASE.append(abs(pred_score - gt_score))
                else:
                    ValueError(f'Found Labels but incorrect processing {p}: labels= {labels}')

            else:
                ValueError(f'No label file found for {p}')


        # Write predicted images if requested
        if write:
            os.makedirs(write_dir, exist_ok=True)
            pred_path = osp.join(write_dir, img_name + '_pred' + img_ext)
            img_with_pred = draw(img.copy(), pred[:, :2], cfg, circles=False, score=pred_score)
            if test:
                gt_path = osp.join(write_dir, img_name + '_gt' + img_ext)
                img_with_gt = draw(img.copy(), gt_xy[:, :2], cfg, circles=False, score=gt_score)
                if fail_cases & ASE[-1] > 0:
                    fail_dir = osp.join(write_dir, 'fail_cases')
                    os.makedirs(fail_dir, exist_ok=True)
                    gt_path = osp.join(fail_dir, img_name + '_gt' + img_ext)
                    pred_path = osp.join(fail_dir, img_name + '_fail' + img_ext)   
                    print
                print(f'Writing GT: {gt_path}')
                cv2.imwrite(gt_path, img_with_gt)
            print(f'Writing Pred: {pred_path}')
            cv2.imwrite(pred_path, img_with_pred)
            
    # Calculate metrics
    fps = (len(img_paths) - 1) / (time() - start_time)
    print(f'FPS: {fps:.2f}')
    
    stats = pd.DataFrame()
    stats['img_paths'] = img_paths
    stats['preds'] = [pred.flatten().tolist() for pred in preds]
    stats['estimated'] = estimated

    if test and ASE:
        ASE = np.array(ASE)
        PCS = len(ASE[ASE == 0]) / len(ASE) * 100
        MASE = np.mean(ASE)
        stats['gt_xys'] = [gt.flatten().tolist() for gt in gt_xys]
        stats['ASE']= pd.Series(ASE) if len(ASE) > 0 else pd.Series([None] * len(img_paths))
        print(f'Percent Correct Score (PCS): {PCS:.1f}%')
        print(f'Mean Absolute Score Error (MASE): {MASE:.2f}')

    stats.to_csv(osp.join(write_dir, 'test_results.csv'), index=False)

    with open(osp.join(write_dir, 'test_results.csv'), 'a') as f:
        f.write(f"# FPS: {fps:.2f}\n")
        if test:
            f.write(f"# PCS: {PCS:.2f}%\n")
            f.write(f"# MASE: {MASE:.2f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    parser.add_argument('-pt', '--bestpt', default='first_run/train/weights/best.pt')
    
    # if arguments not provided prompt user to provide them
    img_folder = input("Enter the folder path to the images: ")
    if not osp.exists(img_folder) or img_folder == '':
        print("Invalid folder path using default: dataset/holo_v1/test/images")
        img_folder = 'dataset/holo_v1/test/images'

    test = input("Do the images have corresponding labels? (y/n): ").lower() == 'y'
    write = input("Do you want to write predicted images? (y/n): ").lower() == 'y'
    fail_cases = input("Do you want to write failed cases? (y/n): ").lower() == 'y'
    
    args = parser.parse_args()
    
    # Load model and config
    model = YOLO(args.bestpt)
    cfg_path = osp.join('configs', args.cfg + '.yaml')
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg
    
    # Run inference
    batch_inference(model, img_folder, cfg, test, write, fail_cases)