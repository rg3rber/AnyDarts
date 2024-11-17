import argparse
import os
import os.path as osp
import cv2
import numpy as np

def crop_single_image(img_path, output_size=800, output_dir=None):
    """
    Crop a single dartboard image to a square using direct OpenCV operations.
   
    Args:
        img_path: Path to input image
        output_size: Size of output image (default: 800)
        output_dir: Directory to save cropped image (default: same as input)
    """
    print(f"Processing image: {img_path}")
   
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
       
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
   
    # Get image dimensions
    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")
   
    # Calculate the crop dimensions
    size = min(h, w)
   
    # Calculate crop coordinates to center the image
    x = (w - size) // 2
    y = (h - size) // 2
   
    # Perform the crop
    print(f"Cropping to {size}x{size}")
    cropped = img[y:y+size, x:x+size]
   
    # Resize to desired size
    if output_size != size:
        print(f"Resizing to {output_size}x{output_size}")
        cropped = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
   
    # Prepare output path
    if output_dir is None:
        output_dir = osp.join(osp.dirname(img_path), 'cropped')
    os.makedirs(output_dir, exist_ok=True)
   
    # Create output filename
    base_name = osp.splitext(osp.basename(img_path))[0]
    output_path = osp.join(output_dir, f"{base_name}_cropped_{output_size}x{output_size}.jpg")
   
    # Save the image
    cv2.imwrite(output_path, cropped)
    print(f"Saved cropped image to: {output_path}")
   
    return output_path

def batch_crop_images(input_dir, output_size=800, output_dir=None):
    """
    Crop all images in a directory to a square and resize to a specified size.
   
    Args:
        input_dir: Directory containing the input images
        output_size: Size of output images (default: 800)
        output_dir: Directory to save cropped images (default: <input_dir>/cropped)
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
   
    if output_dir is None:
        output_dir = osp.join(input_dir, 'cropped')
    os.makedirs(output_dir, exist_ok=True)
   
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = osp.join(input_dir, filename)
            crop_single_image(input_path, output_size, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input directory containing images')
    parser.add_argument('-s', '--size', type=int, default=800, help='Output image size')
    parser.add_argument('-o', '--output-dir', help='Output directory (optional)')
    args = parser.parse_args()
   
    try:
        batch_crop_images(args.input, args.size, args.output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()