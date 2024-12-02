import argparse
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def change_contrast(image, factor=1.5):
    """Adjust image contrast."""
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    return np.array(enhancer.enhance(factor))

def low_pass_filter(image, kernel_size=5):
    """Apply low-pass filter for blurring."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def to_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def change_brightness(image, factor=0.5):
    """Adjust image brightness."""
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    return np.array(enhancer.enhance(factor))

def change_img_hue(image, factor=1.5):
    """Adjust color temperature."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = (hsv[:,:,0] * factor) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def change_color_warmth(image, factor=1.5):
    b, g, r = cv2.split(image)
    b = np.clip(b * (1 * factor), 0, 255).astype(np.uint8)
    r = np.clip(r / factor, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))

def change_sharpness(image, factor=1.5):
    """Adjust image sharpness."""
    enhancer = ImageEnhance.Sharpness(Image.fromarray(image))
    return np.array(enhancer.enhance(factor))

def process_image(input_path, output_folder, transformations):
    """Apply selected transformations to an image."""
    image = cv2.imread(input_path)
    filename = os.path.basename(input_path)
    base_name, ext = os.path.splitext(filename)

    # Save original image
    orig_output_path = os.path.join(output_folder, filename)
    cv2.imwrite(orig_output_path, image)

    # Apply transformations
    for transform in transformations:
        copy = image.copy()
        
        if transform == 'contrast':
            copy = change_contrast(copy)
        elif transform == 'lowpass':
            copy = low_pass_filter(copy)
        elif transform == 'grayscale':
            copy = to_grayscale(copy)
            if len(copy.shape) == 2:
                copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)
        elif transform == 'brightness':
            copy = change_brightness(copy)
        elif transform == 'warmth':
            copy = change_color_warmth(copy)
        elif transform == 'sharpness':
            copy = change_sharpness(copy)
        
        # Save transformed image with transformation name
        transformed_filename = f"{base_name}_{transform}_half{ext}"
        transformed_output_path = os.path.join(output_folder, transformed_filename)
        cv2.imwrite(transformed_output_path, copy)

def main():
    parser = argparse.ArgumentParser(description='Apply image distortions')
    parser.add_argument('-i', '--input_folder', required=True, help='Input image folder')
    parser.add_argument('-o', '--output_folder', required=True, help='Output image folder')
    parser.add_argument('-t', '--transformations', nargs='+', 
                        choices=['contrast', 'lowpass', 'grayscale', 'brightness', 'warmth', 'sharpness'], 
                        default=[], help='Transformations to apply')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            input_path = os.path.join(args.input_folder, filename)
            process_image(input_path, args.output_folder, args.transformations)
            print(f'Processed: {filename} with transformations: {args.transformations}')

if __name__ == '__main__':
    main()