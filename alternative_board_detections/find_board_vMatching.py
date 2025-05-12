import cv2
import numpy as np
import argparse
import os
import time
import os.path as osp
import matplotlib.pyplot as plt



def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

import cv2
import numpy as np

def find_board_vMatching(query_image_path, reference_image_path, reference_size=800):
    """
    Find a dartboard in a query image by matching it with a reference dartboard image using feature matching.
        query_image_path (str): Path to the image containing the dartboard to find.
        reference_image_path (str): Path to the reference dartboard image.
        reference_size (int, optional): Size of the reference dartboard in pixels. Defaults to 800.
            center_point (tuple): The (x, y) coordinates of the detected dartboard center in the query image.
            estimated_size (int): The estimated size of the dartboard in the query image (in pixels).
            annotated_image (numpy.ndarray): The query image annotated with the detected dartboard outline and center.
    Raises:
        ValueError: If one or both images cannot be opened, or if features cannot be detected, or if not enough good matches are found, or if the homography matrix cannot be computed.
    """
    start = time.time()
    # Read images
    query_img = cv2.imread(query_image_path)
    reference_img = cv2.imread(reference_image_path)
    
    if query_img is None or reference_img is None:
        raise ValueError("Could not open one or both images")
        
    # Convert both images to HSV for better color matching
    query_gray_original = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    
    # remove noise
    query_gray = cv2.GaussianBlur(query_gray_original, (5, 5), 0)

    # Initialize feature detector and matcher
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(reference_gray, None)
    kp2, des2 = sift.detectAndCompute(query_gray, None)
    
    if des1 is None or des2 is None:
        raise ValueError("Could not detect features in one or both images")
    

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
 
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(reference_gray, kp1, query_gray, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.imshow(img3),plt.show()
        
    """
    # FLANN matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 10:
        raise ValueError("Not enough good matches found")
    
    # Get corresponding points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        raise ValueError("Could not find homography matrix")
    
    # Get reference image dimensions
    h, w = reference_gray.shape[:2]
    
    # Transform corners of reference image
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    # Calculate center and size
    center_x = np.mean(transformed_corners[:, 0, 0])
    center_y = np.mean(transformed_corners[:, 0, 1])
    center_point = (int(center_x), int(center_y))
    
    # Calculate size
    width = np.linalg.norm(transformed_corners[3, 0] - transformed_corners[0, 0])
    height = np.linalg.norm(transformed_corners[1, 0] - transformed_corners[0, 0])
    avg_size = (width + height) / 2
    
    # Calculate real-world size based on reference
    pixels_per_unit = reference_size / avg_size
    estimated_size = int(avg_size * pixels_per_unit)
    
    # Draw results on image
    result_img = query_img.copy()
    
    # Draw dartboard outline
    cv2.polylines(result_img, [np.int32(transformed_corners)], True, (0, 255, 0), 3)
    
    # Draw center point
    cv2.circle(result_img, center_point, 5, (0, 0, 255), -1)
    
    # Add size information
    text = f"Size: {estimated_size}px"
    cv2.putText(result_img, text, (center_point[0] + 10, center_point[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print("Time taken:", time.time() - start)
    """
    return None
    #return center_point, estimated_size, result_img

def visualize_matches(query_image_path, reference_image_path, num_matches=10):
    """
    Visualize the feature matches between reference and query images.
    
    Args:
        query_image_path (str): Path to the query image
        reference_image_path (str): Path to the reference image
        num_matches (int): Number of best matches to draw
    """
    # Read images
    query_img = cv2.imread(query_image_path)
    reference_img = cv2.imread(reference_image_path)
    
    # Convert to HSV
    query_hsv = cv2.cvtColor(query_img, cv2.COLOR_BGR2HSV)
    reference_hsv = cv2.cvtColor(reference_img, cv2.COLOR_BGR2HSV)
    
    # Initialize SIFT and find keypoints
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(reference_hsv, None)
    kp2, des2 = sift.detectAndCompute(query_hsv, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])
    
    # Sort matches by distance
    good_matches = sorted(good_matches, key=lambda x: x[0].distance)
    
    # Draw top N matches
    match_img = cv2.drawMatchesKnn(
        reference_hsv, kp1, query_hsv, kp2, 
        good_matches[:num_matches], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return match_img

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
    parser.add_argument('-i', '--input', help='Path to input image.', default='images/testboard.jpg')
    args = parser.parse_args()
    board_params = find_board_vMatching(args.input, 'images/reference_board_blur_gray.png', 800)
    print("center point:", board_params[0])
    print("estimated size:", board_params[1])
    cv2.imshow('Result', board_params[2])
    cv2.waitKey()
    #visualize_matches(args.input, 'images/gray_ref_blur.png', num_matches=3)