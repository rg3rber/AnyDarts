import argparse
import os
import cv2
from HelperFunctions import draw, bboxes_to_xy
import numpy as np
from yacs.config import CfgNode

def draw_labels(cfg, dataset_path, output_path):

    bbox_size = cfg.model.bbox_size

    train_output_path = os.path.join(output_path, 'train')
    val_output_path = os.path.join(output_path, 'val')
    
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)

    train_labels_path = os.path.join(dataset_path, 'train', 'labels')
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    val_labels_path = os.path.join(dataset_path, 'val', 'labels')
    val_images_path = os.path.join(dataset_path, 'val', 'images')

    if not os.path.isdir(train_labels_path) or not os.path.isdir(train_images_path) or not os.path.isdir(val_labels_path) or not os.path.isdir(val_images_path):
        print(f"Missing either train or val directory inside {dataset_path} not exist")
        return
    
    # Draw train images
    for label_file in os.listdir(train_labels_path):
        with open(os.path.join(train_labels_path, label_file), 'r') as f:
            lines = f.readlines()
            xywh = np.empty((0, 4), dtype=float)
            classes = np.empty((0,), dtype=int)
            for line in lines:
                class_id, x_center, y_center, w, h = line.split(' ')
                x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
                class_id = int(class_id)
                if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                    print(f"Invalid label in {label_file}")
                    raise ValueError("Invalid label")
                xywh = np.vstack((xywh, [x_center, y_center, w, h]))
                classes = np.append(classes, [class_id])
            bboxes = np.concatenate((xywh, classes[:, None]), axis=1)
            xy, _ = bboxes_to_xy(bboxes, cfg.model.max_darts)
            xy = xy[xy[:, 2] != 0]  # Drop rows where the third column is 0
            xy = xy[:, :2]  # Take only the first two columns
            
        image_name = os.path.splitext(label_file)[0]
        image_path = os.path.join(train_images_path, image_name+'.jpg')
        img = cv2.imread(image_path)
        img = draw(img, xy, cfg, False, True)
        cv2.imwrite(os.path.join(train_output_path, image_name + '_annotated.jpg' ), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check YOLO format labels by drawing them on images')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Path to YOLO dataset')
    parser.add_argument('--cfg', '-c', default='configs/holo_v1.yaml', help='Path to config file')
    parser.add_argument('--output', '-o', default='dataset/check_labels_from_yolo_ds', help='Output directory for annotated images')

    args = parser.parse_args()
    dataset_path = args.dataset
    output_path = args.output
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(args.cfg)

    if not os.path.isdir(dataset_path):
        print(f"Dataset path {dataset_path} does not exist")

    draw_labels(cfg, dataset_path, output_path)
    
    
    