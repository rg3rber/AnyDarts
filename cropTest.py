import argparse
import multiprocessing as mp
from dataset.annotate import crop_board
import os
import os.path as osp
import cv2
import pandas as pd

def crop(img_path, write_path, bbox, size=480, overwrite=False):
    if osp.exists(write_path) and not overwrite:
        print(write_path, 'already exists')
        return
    
    os.makedirs(osp.dirname(write_path), exist_ok=True)
    
    crop = crop_board(img_path, bbox)
    if size != 'full':
        crop = cv2.resize(crop, (size, size))
    cv2.imwrite(write_path, crop)
    print('Wrote', write_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--labels-path', default='dataset/labels.pkl')
    parser.add_argument('-ip', '--image-path', default='dataset/images')
    parser.add_argument('-s', '--size', nargs='+', default=['480'])
    args = parser.parse_args()

    labels = pd.read_pickle(args.labels_path)

    for size in args.size:
        if size != 'full':
            size = int(size)
        
        write_prefix = osp.join(osp.dirname(args.image_path), 'cropped_images', str(size))
        os.makedirs(write_prefix, exist_ok=True)
        
        print('Read path:', args.image_path)
        print('Write path:', write_prefix)

        # Get all image paths in the input directory
        img_paths = []
        write_paths = []
        bboxes = []
        for root, _, files in os.walk(args.image_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img_path = osp.join(root, file)
                    
                    # Find corresponding bbox from labels
                    matching_labels = labels[labels.img_name == file]
                    if not matching_labels.empty:
                        bbox = matching_labels.bbox.values[0]
                        
                        # Create write path maintaining folder structure
                        relative_path = osp.relpath(root, args.image_path)
                        write_path = osp.join(write_prefix, relative_path, file)
                        
                        img_paths.append(img_path)
                        write_paths.append(write_path)
                        bboxes.append(bbox)

        sizes = [size for _ in range(len(bboxes))]

        with mp.Pool(mp.cpu_count()) as p:
            p.starmap(crop, list(zip(img_paths, write_paths, bboxes, sizes)))

if __name__ == '__main__':
    main()