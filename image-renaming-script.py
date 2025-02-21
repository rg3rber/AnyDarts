import argparse
import os
from datetime import datetime
from PIL import Image, ExifTags

def get_image_creation_date(image_path):
    """
    Extract image creation date from various metadata sources.
    Falls back to file creation/modification time if no metadata is available.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Try to get EXIF data
        exif = img._getexif()
        
        if exif:
            # Dictionary of EXIF tags for creation/modification dates
            date_tags = {
                36867: 'DateTimeOriginal',  # Capture date
                36868: 'DateTimeDigitized', # Digitization date
                306:   'DateTime'           # Modification date
            }
            
            for tag_id, tag_name in date_tags.items():
                if tag_id in exif:
                    try:
                        # Parse date string
                        return datetime.strptime(exif[tag_id], '%Y:%m:%d %H:%M:%S')
                    except (ValueError, TypeError):
                        continue
    except Exception:
        pass
    
    # Fallback to file system timestamps
    try:
        # Prefer creation time, fall back to modification time
        stat = os.stat(image_path)
        creation_time = stat.st_ctime if hasattr(stat, 'st_ctime') else stat.st_mtime
        return datetime.fromtimestamp(creation_time)
    except Exception:
        # Last resort: use current time
        return datetime.now()

def rename_images_in_dataset(base_path='images', start_id=1):
    """
    Rename images across all subfolders in the base path.
    Images are renamed with format: IMG_id_{folder_name}.ext
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'}
    
    # First pass: collect all image paths with their creation dates
    image_info = []
    for root, _, files in os.walk(base_path):
        for filename in files:
            # Check if file is an image
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(root, filename)
                try:
                    creation_date = get_image_creation_date(full_path)
                    # Get the folder name (last directory in path)
                    folder_name = os.path.basename(root)
                    image_info.append((full_path, creation_date, folder_name))
                except Exception as e:
                    print(f"Could not process {full_path}: {e}")
    
    # Sort images by creation date
    image_info.sort(key=lambda x: x[1])
    
    # Second pass: rename images
    for global_id, (image_path, creation_date, folder_name) in enumerate(image_info, start=start_id):
        # Extract file extension
        file_ext = os.path.splitext(image_path)[1]
        
        # Create new filename with folder name
        new_filename = f"IMG_{global_id}_{folder_name}{file_ext}"
        #file_name = os.path.basename(image_path)
        #new_filename = f"{folder_name}_{file_name}"
        
        # Get directory of the current image
        directory = os.path.dirname(image_path)
        
        # Full path for new filename
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(image_path, new_path)
        print(f"Renamed: {image_path} -> {new_path}")
    
    print(f"Processed {len(image_info)} images.")

def main(folder_path, start_id):
    # Specify the base path where your image folders are located
    base_path = folder_path
    
    # Confirm before running
    print(f"This script will rename all images in {base_path} and its subfolders. Starting ID: {start_id}")
    confirm = input("Are you sure you want to proceed? (yes/no): ").lower()
    
    if confirm == 'yes':
        rename_images_in_dataset(base_path, start_id)
        print("Image renaming completed.")
    else:
        print("Operation cancelled.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rename images in a dataset')
    parser.add_argument('-f', '--folder-path', default='dataset/images', help='Path to the folder containing images')
    parser.add_argument('-s', '--start-id', default=1, type=int, help='Starting ID for renaming')
    args = parser.parse_args()
    
    main(args.folder_path, args.start_id)