import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_to_grayscale_3ch(input_path):
    """
    Convert an image to grayscale but keep it as 3 channels.
    """
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Warning: Could not read image {input_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert back to 3 channels
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return gray_3ch

def process_yolo_dataset(root_dir):
    """
    Process all images in a YOLO dataset structure.
    
    Expected structure:
    root_dir/
        train/
            images/
            labels/
        val/
            images/
            labels/
        test/
            images/
            labels/
    """
    # Define the subdirectories to process
    subdirs = ["train", "val", "test"]
    
    for subdir in subdirs:
        img_dir = os.path.join(root_dir, subdir, "images")
        
        if not os.path.exists(img_dir):
            print(f"Directory not found: {img_dir}")
            continue
            
        print(f"Processing images in {img_dir}...")
        
        # Get all image files
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Create a backup directory if it doesn't exist
        backup_dir = os.path.join(root_dir, subdir, "images_original")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Process each image
        for img_file in tqdm(img_files):
            img_path = os.path.join(img_dir, img_file)
            
            # Create a backup of the original image
            backup_path = os.path.join(backup_dir, img_file)
            if not os.path.exists(backup_path):
                os.rename(img_path, backup_path)
            
            # Convert to grayscale and save
            gray_img = convert_to_grayscale_3ch(backup_path)
            if gray_img is not None:
                cv2.imwrite(img_path, gray_img)

if __name__ == "__main__":
    # Define the root directory of your YOLO dataset
    data_root = "data"  # Change this if your root directory is different
    
    # Process the dataset
    process_yolo_dataset(data_root)
    
    print("Processing complete. Original images are backed up in 'images_original' folders.")