import pandas as pd
import os
import os.path as osp
from typing import List, Tuple
import argparse
from yacs.config import CfgNode

def create_yolo_annotation(points: List[List[float]], output_path: str, bbox_size: float) -> None:
    """
    Create YOLO format annotation file from a list of points.
    First 4 points are calibration points (class 0-3), remaining points are darts (class 4).
    
    Args:
        points: List of [x, y] center coordinates (already normalized)
        output_path: Path where to save the txt file
        bbox_size: Size of the bounding box (normalized, same for width and height)
    """
    with open(output_path, 'w') as f:
        # Process calibration points (first 4 points)
        for class_id, point in enumerate(points[:4]):
            x_center, y_center = point
            # Write in YOLO format: class x_center y_center width height
            f.write(f"{class_id} {x_center} {y_center} {bbox_size:.6f} {bbox_size:.6f}\n")
        
        # Process dart points (remaining points)
        for point in points[4:]:
            x_center, y_center = point
            f.write(f"4 {x_center} {y_center} {bbox_size:.6f} {bbox_size:.6f}\n")

def convert_annotations(pkl_path: str, output_dir: str, bbox_size: float) -> None:
    """
    Convert pickle file annotations to YOLO format txt files.
    
    Args:
        pkl_path: Path to the pickle file containing annotations
        output_dir: Directory where to save the txt files
        bbox_size: Size of the bounding box (normalized, same for width and height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the pickle file
    df = pd.read_pickle(pkl_path)
    
    # Process each row
    for _, row in df.iterrows():
        # Get image name and points
        img_folder = row['img_folder']
        img_name = row['img_name']
        points = row['xy']
        
        # Skip if there are no points (although this shouldn't happen in your case)
        if not points:
            continue
            
        # Create output path (change extension from jpg to txt)
        # add img folder for deepdarts images
        # output_path = os.path.join(output_dir, f"{img_folder}_{os.path.splitext(img_name)[0]}" + '.txt')
        
        output_path = os.path.join(output_dir, f"{img_name.split('.')[0]}" + '.txt')

        # Create YOLO annotation file
        create_yolo_annotation(points, output_path, bbox_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--labels-path', default='labels.pkl')
    parser.add_argument('-o', '--output-path', default='images/labels')
    parser.add_argument('-c', '--config', default='configs/any_v2.yaml')

    args = parser.parse_args()

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(args.config)

    output_dir = osp.join(args.output_path, osp.basename(osp.splitext(args.labels_path)[0]))

    convert_annotations(args.labels_path, output_dir, cfg.model.bbox_size)
    print(f"Conversion completed. Label files saved in {args.output_path}")
