import albumentations as A
import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_yolo_labels(label_path):
    """Parse YOLO format labels from a file."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    classes = []
    bboxes = []
    
    for line in lines:
        line = line.strip().split()
        if len(line) == 5:
            class_id = int(line[0])
            x_center, y_center, width, height = [float(x) for x in line[1:]]
            classes.append(class_id)
            bboxes.append([x_center, y_center, width, height])
    
    return classes, bboxes

def save_yolo_labels(label_path, classes, bboxes):
    """Save bounding boxes in YOLO format."""
    with open(label_path, 'w') as f:
        for class_id, bbox in zip(classes, bboxes):
            x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def apply_augmentations(img, classes, bboxes, aug_config):
    """Apply augmentations based on the configuration."""
    h, w = img.shape[:2]
    
    # Separate calibration points (classes 0-3) and darts (class 4)
    calib_indices = [i for i, cls in enumerate(classes) if cls < 4]
    dart_indices = [i for i, cls in enumerate(classes) if cls >= 4]
    
    # Make sure inputs are valid and handle empty cases
    if not dart_indices:
        return img.copy(), classes.copy(), bboxes.copy(), []
    
    dart_classes = [classes[i] for i in dart_indices]
    dart_bboxes = [bboxes[i] for i in dart_indices]
    
    # Make sure all bounding boxes are valid
    dart_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in dart_bboxes]
    
    augmented_img = img.copy()
    augmented_classes = classes.copy()
    augmented_bboxes = bboxes.copy()
    
    # Make sure all initial bounding boxes are valid
    augmented_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented_bboxes]
    
    # TODO remove
    augs_applied = []
    
    # Apply random augmentations based on probabilities
    if np.random.uniform() < aug_config['overall_prob']:
        # Horizontal flip for darts only
        if np.random.uniform() < aug_config['flip_lr_prob']:
            try:
                flip_transform = A.Compose([
                    A.HorizontalFlip(p=1.0)
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['classes']))
                
                augmented = flip_transform(
                    image=augmented_img,
                    bboxes=dart_bboxes,
                    classes=dart_classes
                )
                
                augmented_img = augmented['image']
                
                # Update dart bboxes
                for i, idx in enumerate(dart_indices):
                    if i < len(augmented['bboxes']):
                        augmented_bboxes[idx] = np.clip(augmented['bboxes'][i], 0.0, 1.0)
                
                augs_applied.append('flip_lr')
            except Exception as e:
                print(f"Error in horizontal flip: {e}")
                # Continue with the next transformation
        
        # Vertical flip for darts only
        if np.random.uniform() < aug_config['flip_ud_prob']:
            try:
                flip_transform = A.Compose([
                    A.VerticalFlip(p=1.0)
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['classes']))
                
                augmented = flip_transform(
                    image=augmented_img,
                    bboxes=dart_bboxes,
                    classes=dart_classes
                )
                
                augmented_img = augmented['image']
                
                # Update dart bboxes
                for i, idx in enumerate(dart_indices):
                    if i < len(augmented['bboxes']):
                        augmented_bboxes[idx] = np.clip(augmented['bboxes'][i], 0.0, 1.0)
                
                augs_applied.append('flip_ud')
            except Exception as e:
                print(f"Error in vertical flip: {e}")
                # Continue with the next transformation
        
        # Rotation (large angles) - only for darts
        if np.random.uniform() < aug_config['rot_prob']:
            try:
                angles = np.arange(-180, 180, step=aug_config['rot_step'])
                angle = angles[np.random.randint(len(angles))]
                
                rotate_transform = A.Compose([
                    A.Rotate(limit=(angle, angle), p=1.0)
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['classes']))
                
                augmented = rotate_transform(
                    image=augmented_img,
                    bboxes=dart_bboxes,
                    classes=dart_classes
                )
                
                augmented_img = augmented['image']
                
                # Update dart bboxes
                for i, idx in enumerate(dart_indices):
                    if i < len(augmented['bboxes']):
                        augmented_bboxes[idx] = np.clip(augmented['bboxes'][i], 0.0, 1.0)
                
                augs_applied.append(f'rot_{angle}')
            except Exception as e:
                print(f"Error in large rotation: {e} at image {img}")
                # Continue with the next transformation
        
        # Small rotation - for all points
        if np.random.uniform() < aug_config['rot_small_prob']:
            try:
                angle = np.random.uniform(-aug_config['rot_small_max'], aug_config['rot_small_max'])
                
                # Ensure bboxes are valid before transform
                all_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented_bboxes]
                all_classes = augmented_classes
                
                rotate_transform = A.Compose([
                    A.Rotate(limit=(angle, angle), p=1.0)
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['classes']))
                
                augmented = rotate_transform(
                    image=augmented_img,
                    bboxes=all_bboxes,
                    classes=all_classes
                )
                
                augmented_img = augmented['image']
                augmented_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented['bboxes']]
                augmented_classes = augmented['classes']
                
                augs_applied.append(f'rot_small_{angle}')
            except Exception as e:
                print(f"Error in small rotation: {e} at image {img}")
                # Continue with the next transformation
        
        # Jitter (translation)
        if np.random.uniform() < aug_config['jitter_prob']:
            try:
                jitter = aug_config['jitter_max'] * min(h, w)
                tx = np.random.uniform(0, jitter)
                ty = np.random.uniform(0, jitter)
                
                # Ensure bboxes are valid before transform
                all_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented_bboxes]
                all_classes = augmented_classes
                
                shift_transform = A.Compose([
                    A.ShiftScaleRotate(
                        shift_limit_x=(-tx / w, tx / w),
                        shift_limit_y=(-ty / h, ty / h),
                        scale_limit=0,
                        rotate_limit=0,
                        p=1.0
                    )
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['classes']))
                
                augmented = shift_transform(
                    image=augmented_img,
                    bboxes=all_bboxes,
                    classes=all_classes
                )
                
                augmented_img = augmented['image']
                augmented_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented['bboxes']]
                augmented_classes = augmented['classes']
                
                augs_applied.append(f'jitter_{tx}_{ty}')
            except Exception as e:
                print(f"Error in jitter: {e} at image {img}")
                # Continue with the next transformation
        
        # Warp (affine transformation)
        if np.random.uniform() < aug_config['warp_prob']:
            try:
                warp_scale = aug_config['warp_rho'] / 100.0  # Scale down to make it less aggressive
                
                # Ensure bboxes and classes have the same length and are valid
                if len(augmented_bboxes) != len(augmented_classes):
                    min_len = min(len(augmented_bboxes), len(augmented_classes))
                    augmented_bboxes = augmented_bboxes[:min_len]
                    augmented_classes = augmented_classes[:min_len]
                
                # Ensure bboxes are valid before transform
                augmented_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented_bboxes]
                
                warp_transform = A.Compose([
                    A.Affine(
                        shear={"x": (-30*warp_scale, 30*warp_scale), "y": (-30*warp_scale, 30*warp_scale)},
                        p=1.0
                    )
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['classes']))
                
                augmented = warp_transform(
                    image=augmented_img,
                    bboxes=augmented_bboxes,
                    classes=augmented_classes
                )
                
                augmented_img = augmented['image']
                augmented_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented['bboxes']]
                augmented_classes = augmented['classes']
                
                augs_applied.append(f'warp_{warp_scale}')
            except Exception as e:
                print(f"Error in warp: {e}, at image {img}")
                # Continue with the next transformation
    
    # Final sanity check
    augmented_bboxes = [np.clip(bbox, 0.0, 1.0) for bbox in augmented_bboxes]
    
    return augmented_img, augmented_classes, augmented_bboxes, augs_applied

def generate_augmentations(dataset_path, output_path, aug_config, multiplier=4):
    """Generate augmentations for all images in the dataset."""
    # Create output directories if they don't exist
    train_img_out = os.path.join(output_path, 'train', 'images')
    train_label_out = os.path.join(output_path, 'train', 'labels')
    val_img_out = os.path.join(output_path, 'val', 'images')
    val_label_out = os.path.join(output_path, 'val', 'labels')
    
    os.makedirs(train_img_out, exist_ok=True)
    os.makedirs(train_label_out, exist_ok=True)
    os.makedirs(val_img_out, exist_ok=True)
    os.makedirs(val_label_out, exist_ok=True)
    
    # Process train set
    process_dataset_split(
        os.path.join(dataset_path, 'train'),
        train_img_out,
        train_label_out,
        aug_config,
        multiplier
    )
    
    # Process validation set (no augmentations, just copy)
    process_dataset_split(
        os.path.join(dataset_path, 'val'),
        val_img_out,
        val_label_out,
        aug_config,
        multiplier=1,  # No augmentation for val set
        augment=False
    )

def process_dataset_split(split_path, img_out_path, label_out_path, aug_config, multiplier=4, augment=True):
    """Process a dataset split (train or val)."""
    img_dir = os.path.join(split_path, 'images')
    label_dir = os.path.join(split_path, 'labels')

    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + 
                  glob.glob(os.path.join(img_dir, '*.JPG')) +
                  glob.glob(os.path.join(img_dir, '*.png')) +
                  glob.glob(os.path.join(img_dir, '*.PNG')))
    
    print(f"Processing {len(img_paths)} images in {split_path}")

    # Augmentations applied per image: 
    augs_applied_map = [] # TODO remove
    
    for i, img_path in enumerate(tqdm(img_paths)):
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(label_dir, f"{img_name}.txt")

        if i / len(img_paths) > 0.58 and i / len(img_paths) < 0.60:
            print(f"Processing image at {i/len(img_paths)*100:.2f}%: {img_path}")
        
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_filename}")
            continue
        
        # Read image and labels
        img = cv2.imread(img_path)
        classes, bboxes = parse_yolo_labels(label_path)
        
        # Copy the original image
        output_img_path = os.path.join(img_out_path, img_filename)
        output_label_path = os.path.join(label_out_path, f"{img_name}.txt")
        
        cv2.imwrite(output_img_path, img)
        save_yolo_labels(output_label_path, classes, bboxes)

        # Generate augmentations
        if augment and multiplier > 1:
            for i in range(1, multiplier):
                aug_img, aug_classes, aug_bboxes, augs_applied = apply_augmentations(img, classes, bboxes, aug_config) # TODO remove
                augs_applied_str = '_'.join(augs_applied) # TODO remove
                aug_img_name = f"{img_name}_aug{i}{os.path.splitext(img_filename)[1]}"
                augs_applied_map.append([aug_img_name, augs_applied_str]) # TODO remove
                aug_label_name = f"{img_name}_aug{i}.txt"
                
                aug_img_path = os.path.join(img_out_path, aug_img_name)
                aug_label_path = os.path.join(label_out_path, aug_label_name)
                
                cv2.imwrite(aug_img_path, aug_img)
                save_yolo_labels(aug_label_path, aug_classes, aug_bboxes)

    # # TODO remove
    # with open(os.path.join(label_out_path, f"_imgsAugLog.txt"), 'w') as f:
    #     for aug in augs_applied_map:
    #         f.write(f"{aug[0]}: {aug[1]}\n")
    #     print(f"wrote augmentation log to {label_out_path}/_imgsAugLog.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmentations for YOLO dataset")
    parser.add_argument("--dataset", '-d', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output", '-o', type=str, required=True, help="Path to the output directory")
    parser.add_argument("--times", '-t',type=int, default=4, help="Number of augmentations per image")
    args = parser.parse_args()
    
    # DeepDarts Augmentation configuration
    aug_config = {
        'overall_prob': 0.8,
        'flip_lr_prob': 0.5,
        'flip_ud_prob': 0.5,
        'rot_prob': 0.5,
        'rot_step': 36,
        'rot_small_prob': 0.5,
        'rot_small_max': 2,
        'jitter_prob': 0.5,
        'jitter_max': 0.02,
        'cutout_prob': 0,
        'warp_prob': 0.5,
        'warp_rho': 2
    }
    
    generate_augmentations(args.dataset, args.output, aug_config, args.times)
    print(f"Augmentations generated successfully. Output saved to {args.output}")