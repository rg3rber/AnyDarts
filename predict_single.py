import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
import cv2
import numpy as np
from dataset.annotate import draw, get_dart_scores
import sys

def build_and_load_model(cfg):
    """
    Build the YOLOv4 model and load the pre-trained weights.
    """
    yolo = build_model(cfg)

    # Load weights from the checkpoint
    try:
        print(f"Loading weights from: {osp.join('models', cfg.model.name)}")
        yolo.model.load_weights(osp.join('models', cfg.model.name, 'weights.index'), by_name=True)
        yolo.model.load_weights(osp.join('models', cfg.model.name, 'weights.data-00000-of-00001'), by_name=True)
    except Exception as e:
        raise ValueError(f"Error loading weights: {str(e)}")

    return yolo

def crop_image(img, target_size=800):
    """Crop the image to a square of target_size x target_size."""
    print(f"Cropping image to {target_size}x{target_size}")
    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    size = min(h, w)
    
    # Calculate crop coordinates to center the image
    x = (w - size) // 2
    y = (h - size) // 2
    
    # Crop and resize
    cropped = img[y:y+size, x:x+size]
    if size != target_size:
        cropped = cv2.resize(cropped, (target_size, target_size))
    print(f"Successfully cropped and resized image")
    return cropped

def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy

def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        print('Missed more than 1 calibration point')
    return xy

def predict_single_image(image_path, yolo, cfg, output_path=None, max_darts=3):
    """
    Make a prediction on a single image.
    
    Args:
        image_path: Path to the input image
        yolo: YOLOv4 model instance
        cfg: Configuration object
        output_path: Path to save the annotated image (optional)
        max_darts: Maximum number of darts to detect
    """
    print(f"\nProcessing image: {image_path}")
    weights_path = osp.join('models', cfg.model.name, 'weights')
    #yolo = build_and_load_model(cfg, weights_path)

    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read and crop image
    print("Reading image...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    #img = crop_image(img, cfg.model.input_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Make prediction
    print("Making prediction...")
    bboxes = yolo.predict(img_rgb)
    print(f"Found {len(bboxes)} bounding boxes")
    
    xy = bboxes_to_xy(bboxes, max_darts)
    
    # Calculate scores
    print("Calculating scores...")
    dart_scores = get_dart_scores(xy[:, :2], cfg, numeric=True)
    total_score = sum(dart_scores)
    
    # Draw results if output path is specified
    if output_path:
        print(f"Saving annotated image to: {output_path}")
        valid_xy = xy[xy[:, -1] == 1]
        annotated_img = draw(img.copy(), valid_xy[:, :2], cfg, circles=False, score=True)
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_img)
    
    return {
        'xy': xy,
        'scores': dart_scores,
        'total_score': total_score
    }

if __name__ == '__main__':
    try:
        print("Initializing prediction script...")
        from train import build_model
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--cfg', default='deepdarts_d1', help='Config file name')
        parser.add_argument('-i', '--input', required=True, help='Input image path')
        parser.add_argument('-o', '--output', help='Output image path')
        args = parser.parse_args()
        
        # Load config
        print(f"\nLoading configuration from: {args.cfg}")
        cfg_path = osp.join('configs', args.cfg + '.yaml')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
            
        cfg = CN(new_allowed=True)
        cfg.merge_from_file(cfg_path)
        cfg.model.name = args.cfg
        
        # Build and load model
        print("\nBuilding model...")
        
        yolo = build_model(cfg)
        yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)
         
        print("making prediction...")
        # Make prediction
        results = predict_single_image(
            args.input,
            yolo,
            cfg,
            args.output
        )
        
        print("\nResults:")
        print(f"Detected dart scores: {results['scores']}")
        print(f"Total score: {results['total_score']}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)