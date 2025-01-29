import os.path as osp
import os
from ultralytics import YOLO
import cv2
import argparse
from yacs.config import CfgNode
from inference import inference

""" run inference on set of images"""



def batch_inference(model, folder_path, cfg, test=False, write=False):
    images = os.listdir(folder_path)
    img_paths = [osp.join(folder_path, name) for (name) in images]

    gt_xys = []
    pred_xys = []


    total_score = 0
    
    for i, p in enumerate(img_paths):

        img = cv2.imread(p)
        bboxes = model(img, max_det=4+cfg.model.max_darts)

        return total
    for img in images:
        image_path = osp.join(folder_path, img)
        total_score += inference(model, image_path, cfg)
    return total_score
    
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='holo_v1')
    parser.add_argument('-pt', '--bestpt', default='first_run/train/weights/best.pt')

    # if arguments not provided prompt user to provide them
    img_folder = input("Enter the folder path to the images: ")
    if not osp.isdir(img_folder) or img_folder == '':
        print("Invalid folder path using default: dataset/holo_v1/test/images")
        img_folder = 'dataset/holo_v1/test/images'
    test = input("Do the images have corresponding labels? (y/n): ")
    if test == 'y':
        test = True
    else:
        test = False
    write = input("Do you want to write predicted images? (y/n): ")
    if write == 'y':
        write = True
    else:
        write = False


    args = parser.parse_args()

    model = YOLO(args.bestpt)

    cfg_path = osp.join('configs', args.cfg + '.yaml')

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.model.name = args.cfg

    predicted_score = batch_inference(model, img_folder, cfg, test, write)
    print(predicted_score)