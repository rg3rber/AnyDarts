import cv2
import numpy as np
import argparse
import os
import os.path as osp
import random
from random import sample

class Ellipse:
    def __init__(self):
        self.center = None # (x, y)
        self.centerFloat = None # (x, y)
        self.axes = None # (width, height) = (b, a)=(minor,major)
        self.axesFloat = None # (b,a)
        self.angle = None # angle in degrees

def find_board(image):
    """ input: path to image """
    if isinstance(image, str) and osp.isfile(image):
        img = cv2.imread(image)
    else:
        img = image
    
    if img is None:
        print('Could not open or find the image:', image)
        exit(0)

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red and green colors in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    final_mask = cv2.bitwise_or(red_mask, green_mask)
    binary = otsu_thresholding(final_mask)

    resizeMasked, scale_factor_ellipse = proportionalResize(binary, 1000)

    detectedEllipse = Ellipse()
    detectedEllipse.centerFloat, detectedEllipse.axesFloat, detectedEllipse.angle = findEllipse(resizeMasked)
    detectedEllipse.center = (int(detectedEllipse.centerFloat[0]), int(detectedEllipse.centerFloat[1]))
    detectedEllipse.axes = (int(detectedEllipse.axesFloat[0]), int(detectedEllipse.axesFloat[1]))
    
    if detectedEllipse is None:
        print("No ellipse found.")
        return None
    
    resizedEllipse = resize_ellipse(scale_factor_ellipse, detectedEllipse)
    
    bbox = getSquareBboxForEllipse(resizedEllipse.centerFloat, resizedEllipse.axesFloat, h, w) # bbox for cropping on the board with random buffer (0.32-0.35)

    reformatted_bbox = [bbox[0][1], bbox[1][1],  # deepdarts convention: topleft_y, bottomright_y, top_left_x, bottom_right_x
                       bbox[0][0], bbox[1][0]]

    return reformatted_bbox

def resize_ellipse(scale_factor, ellipse):
    resizedEllipse = Ellipse()
    resizedEllipse.center = (int(ellipse.centerFloat[0]/scale_factor), int(ellipse.centerFloat[1]/scale_factor))
    resizedEllipse.centerFloat = (ellipse.centerFloat[0]/scale_factor, ellipse.centerFloat[1]/scale_factor)
    resizedEllipse.axes = (int(ellipse.axesFloat[0]/scale_factor), int(ellipse.axesFloat[1]/scale_factor))
    resizedEllipse.axesFloat = (ellipse.axesFloat[0]/scale_factor, ellipse.axesFloat[1]/scale_factor)
    resizedEllipse.angle = ellipse.angle
    return resizedEllipse

def otsu_thresholding(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, binary_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary_img

def proportionalResize(image, target_size, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    max_side = max(h, w)
    
    if max_side > target_size:
        r = target_size / float(max_side)
        return cv2.resize(image, (0, 0), fx=r, fy=r), r
    return image, 1


def alternatefindEllipse(img, original_img=None, scale_factor=1):
    """
    Kept for reference
    
    input: binary image
    output: params of roated rectangle around ellipse
    center point, axes (b=minor, a=major), angle
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    best_ellipse = None

    for c in contours:
        if c.shape[0] > 5:
            ellipse = cv2.fitEllipse(c)
            (x, y), (w, h), angle = ellipse
            size = w * h
            circularity = w / h
            ellipses.append((ellipse, size, circularity))

    if len(ellipses) == 0:
        return None
    else:
        ellipses.sort(key=lambda x: x[1], reverse=True)

    for i, ellipse in enumerate(ellipses):
        if ellipse[2] > 0.8:
            best_ellipse = ellipse[0] # for now its just the largest ellipse with circularity > 0.8
            break
    if best_ellipse is None:
        best_ellipse = ellipses[0][0] # if no ellipse with circularity > 0.8, return the largest ellipse
    return best_ellipse

def findEllipse(img, img_name=None, original_img=None, scale_factor=1):
    """
    input: binary image
    output: params of roated rectangle around ellipse
    center point, (width, height)=axes=(b,a)=(minor,major), angle
    """  
    img = cv2.GaussianBlur(img.copy(), (3, 3), 0)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    best_ellipse = None

    for c in contours:
        if c.shape[0] > 5:
            rectangle = cv2.fitEllipse(c)
            (x, y), (w, h), angle = rectangle
            size = w * h
            circularity = w / h
            ellipses.append((rectangle, size, circularity))

    if len(ellipses) == 0:
        return None
    else:
        ellipses.sort(key=lambda x: x[1], reverse=True)
    size_of_largest_ellipse = ellipses[0][1]

    for i, ellipse in enumerate(ellipses):
        if ellipse[2] > 0.8:

            if ellipse[1] > 0.5 * size_of_largest_ellipse:
                best_ellipse = ellipse[0] # for now its just the largest ellipse with circularity > 0.8 and it must be at least 50% of the largest ellipse
            break
    if best_ellipse is None:
        best_ellipse = ellipses[0][0] # if no ellipse with circularity > 0.8, return the largest ellipse
    return best_ellipse

def getSquareBboxForEllipse(center, axes, h, w, buffer=None):
    """
    input: center point, axes, image height and width
    output: top left and bottom right corners of a square bounding box with 
    the same center as the ellipse and 25-35% larger than the ellipse
    buffer: optional parameter to set the bounding box to be buffer% larger than the ellipse
    """
  
    scale_factor = 1 + random.uniform(0.3, 0.35) # outer rim is 32% larger than the ellipse
    if buffer is not None:
        scale_factor = 1 + buffer
    long_axis = max(axes) * scale_factor # set the bounding box to be 25-35% larger than the ellipse
    top_left_corner = (max(0, int(center[0] - long_axis/2)), 
                      max(0, int(center[1] - long_axis/2)))
    bottom_right_corner = (min(w, int(center[0] + long_axis/2)), 
                          min(h, int(center[1] + long_axis/2)))
    
    return (top_left_corner, bottom_right_corner)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Board detection using ellipse fitting.')
    parser.add_argument('-i', '--input', help='Path to input image.', 
                       default='boards/myboard/testboard.jpg')
    args = parser.parse_args()
    
    if osp.isdir(args.input):
        for img in os.listdir(args.input):
            if not osp.isdir(osp.join(args.input, img)):
                if (img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')):
                    print("bbox = ", find_board(osp.join(args.input, img)))
                else:
                    print("Skipping non image file: ", img)
    else:
        print("bbox = ", find_board(args.input))