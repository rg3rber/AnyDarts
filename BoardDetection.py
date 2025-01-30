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

def find_board(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)

    """ debug vars"""
    global img_name
    img_name = osp.basename(img_path)

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    """approach 2: (hardcoded color ranges)"""
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

    red_binary = otsu_thresholding(red_mask)
    green_binary = otsu_thresholding(green_mask) #comparing red vs green

    resizeMasked, scale_factor_ellipse = proportionalResize(binary, 1000)

    detectedEllipse = Ellipse()
    detectedEllipse.centerFloat, detectedEllipse.axesFloat, detectedEllipse.angle = findEllipse(resizeMasked, img_path)
    detectedEllipse.center = (int(detectedEllipse.centerFloat[0]), int(detectedEllipse.centerFloat[1]))
    detectedEllipse.axes = (int(detectedEllipse.axesFloat[0]), int(detectedEllipse.axesFloat[1]))
    
    if detectedEllipse is None:
        print("No ellipse found.")
        return None
    
    resizedEllipse = resize_ellipse(scale_factor_ellipse, detectedEllipse)

    inferred_radius = int(np.mean(resizedEllipse.axesFloat)/2) # mean of long and short axes
    
    bbox = getSquareBboxForEllipse(resizedEllipse.centerFloat, resizedEllipse.axesFloat, h, w) # bbox for cropping on the board with random buffer (0.32-0.35)
    approx_full_board_crop = getSquareBboxForEllipse(resizedEllipse.centerFloat, resizedEllipse.axesFloat, h, w, 0.32) # bbox for cropping on the board including the number wire with no buffer

    """ Find Image Center using red mask bullseye """
    
    approx_bbox_length = (approx_full_board_crop[1][0] - approx_full_board_crop[0][0]) # the 451mm diameter of the board 
    bullseye_bounding_box_length = int((approx_full_board_crop[1][0] - approx_full_board_crop[0][0])/5)
    hough_bounding_box_margin = int((approx_full_board_crop[1][0] - approx_full_board_crop[0][0])/3)
    crop_imgToBull = cv2.getRectSubPix(img, (hough_bounding_box_margin, hough_bounding_box_margin), resizedEllipse.centerFloat)

    crop_imgToBBox = cv2.getRectSubPix(img, (approx_bbox_length, approx_bbox_length), resizedEllipse.centerFloat)

    #top_crop_toBull = crop_imgToBull[0:int(hough_bounding_box_margin*0.55), 0:hough_bounding_box_margin] # top 55%. careful crop is y, x
    top_crop = crop_imgToBBox[0:int(approx_bbox_length*0.55), 0:approx_bbox_length]
    #lower_crop_toBull = crop_imgToBull[int(hough_bounding_box_margin*0.45):hough_bounding_box_margin, 0:hough_bounding_box_margin]
    lower_crop = crop_imgToBBox[int(approx_bbox_length*0.45):approx_bbox_length, 0:approx_bbox_length]

    crop_red_binary_toBullseye = cv2.getRectSubPix(red_binary, (bullseye_bounding_box_length, bullseye_bounding_box_length), resizedEllipse.centerFloat)
    #resize_cropToBullseye, scale_factor_red_bullseye = proportionalResize(crop_red_binary_toBullseye, 480)

    #contours_cropped, board_center, radius = find_boardCenterHough(resize_crop_imageToEllipse) #find_boardCenter(resize_crop_imageToEllipse)
    board_center_color, center_axes, center_angle = workingfindEllipse(crop_red_binary_toBullseye)
    remapped_board_center_color = (board_center_color[0]+resizedEllipse.centerFloat[0]-bullseye_bounding_box_length/2, board_center_color[1] + resizedEllipse.centerFloat[1]-bullseye_bounding_box_length/2)
    remapped_board_center_color_Int= (int(remapped_board_center_color[0]), int(remapped_board_center_color[1]))

    """ find exact ellipse using red? mask """

    red_cropToBoard = cv2.getRectSubPix(red_binary, (approx_bbox_length, approx_bbox_length), resizedEllipse.centerFloat)
    red_cropToBoard_center = (approx_bbox_length/2, approx_bbox_length/2)

    ellipse_samples = find_furthest_intersections(red_cropToBoard, red_cropToBoard_center, resizedEllipse.axesFloat, 180) # use approximate ellipse center and axes as reference to find intersections
    #ellipse_samples = np.append(ellipse_samples, np.array([[[0,0]]]), axis=0).astype(np.float32)

    redEllipse = fit_ellipse_ransac(ellipse_samples, image_size=(h,w)) # ransac to be robust to outliers

    if redEllipse is None:
        print("No ellipse found in red mask.")
        return None
    
    remapped_redEllipse = Ellipse()
    remapped_redEllipse.centerFloat = (redEllipse.center[0]+resizedEllipse.centerFloat[0]-approx_bbox_length/2, redEllipse.center[1]+resizedEllipse.centerFloat[1]-approx_bbox_length/2)
    remapped_redEllipse.axesFloat = redEllipse.axesFloat
    remapped_redEllipse.angle = redEllipse.angle
    remapped_redEllipse.center = (int(remapped_redEllipse.centerFloat[0]), int(remapped_redEllipse.centerFloat[1]))
    remapped_redEllipse.axes = (int(remapped_redEllipse.axesFloat[0]), int(remapped_redEllipse.axesFloat[1]))

    cv2.ellipse(img, (resizedEllipse.center, resizedEllipse.axes, resizedEllipse.angle), (0, 255, 0), 1)
    cv2.ellipse(img, (remapped_redEllipse.centerFloat, remapped_redEllipse.axesFloat, resizedEllipse.angle), (255, 0, 255), 2)
    
    cv2.circle(img, remapped_board_center_color_Int, 5, (255, 0, 255), -1)

    cal_pts = compute_cal_pts(remapped_redEllipse, img)

    for pt in cal_pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

    cv2.line(img, (int(cal_pts[0][0]), int(cal_pts[0][1])), (int(cal_pts[1][0]), int(cal_pts[1][1])), (0, 255, 0), 2)
    cv2.line(img, (int(cal_pts[2][0]), int(cal_pts[2][1])), (int(cal_pts[3][0]), int(cal_pts[3][1])), (0, 255, 0), 2)

    cv2.circle(img, remapped_board_center_color_Int, 5, (0, 255, 0), -1)
    cv2.circle(img, remapped_redEllipse.center, 5, (255, 0, 255), -1)
    """ cv2.imshow("Bounding box preview:", img)
    cv2.waitKey() """

    """ # why not use the ellipse found in the first step?
    red_resize, red_scale_factor = proportionalResize(red_binary, 1000)
    cv2.imshow("red", red_resize)
    cv2.waitKey()
    return None
    ellipse_samples = find_furthest_intersections(red_resize, detectedEllipse.centerFloat, detectedEllipse.axesFloat, 720) # use approximate ellipse center and axes as reference to find intersections
    redEllipse = fit_ellipse_to_points(ellipse_samples, center=None, known_axes=None) # before center was board_center_in_red_mask

    if redEllipse is None:
        print("No ellipse found in red mask.")
        return None
    
    resizedRedEllipse = resize_ellipse(red_scale_factor, redEllipse)
    
    cv2.ellipse(img, (resizedRedEllipse.center, resizedRedEllipse.axes, resizedRedEllipse.angle), (255, 0, 255), 2)
    cv2.circle(img, remapped_board_center_color_Int, 5, (255, 0, 255), -1)
    for point in ellipse_samples:
        (x, y) = point[0]
        cv2.circle(img, (int(x/red_scale_factor), int(y/red_scale_factor)), 1, (255, 0, 0), -1)
    cv2.imwrite(osp.join("boards/board_angles/red_ellipse", "New_red_ellipse"+img_name), img)
    print("wrote img: boards/board_angles/green_ellipse/green_ellipse" + img_name) """


    """ Hough lines to find segment lines """


    """  different approach try to rotate the red mask and find the ellipse --------------------- 
    else:
        resize_red, scale_factor_red = proportionalResize(red_binary, 1000)
        remapped_center_color = (int(board_center_color[0]/scale_factor_circle + bbox[0][0]+bullseye_bounding_box_margin), int(board_center_color[1]/scale_factor_circle + bbox[0][1]+bullseye_bounding_box_margin))
        red_center = (int(remapped_center_color[0]*scale_factor_red), int(remapped_center_color[1]*scale_factor_red))
        rotate_red_9 = rotate_image(resize_red, red_center, 18)
        red_board = cv2.bitwise_or(rotate_red_9, resize_red)
        cv2.imshow("red", red_board)
        cv2.waitKey()
        red_ellipse = workingfindEllipse(red_board)
        if red_ellipse is None:
            print("No red ellipse found.")
            return None
        red_center, red_axes, red_angle = red_ellipse
        print("red ellipse = ", red_center, red_axes, red_angle)
        red_bbox = getSquareBboxForEllipse(red_center, red_axes, red_board.shape[0], red_board.shape[1])
        resized_red_bbox = (int(red_bbox[0][0]/scale_factor_red), int(red_bbox[0][1]/scale_factor_red)), \
                   (int(red_bbox[1][0]/scale_factor_red), int(red_bbox[1][1]/scale_factor_red))
        print("red bbox = ", resized_red_bbox)
        cv2.ellipse(img, red_ellipse, (0, 0, 255), 10)
        cv2.rectangle(img, resized_red_bbox[0], resized_red_bbox[1], (0, 0, 255), 10)
        cv2.imshow("red ellipse", img)
        cv2.waitKey() """

    reformatted_bbox = [bbox[0][1], bbox[1][1], 
                       bbox[0][0], bbox[1][0]]
    
    #these lines are for visualization purposes only
    #resized_ellipse = (ellipseCenter[0]/scale_factor, ellipseCenter[1]/scale_factor), (axes[0]/scale_factor, axes[1]/scale_factor), angle

    resizedEllipse.axesFloat = (int(resizedEllipse.axesFloat[0]/scale_factor_ellipse), int(resizedEllipse.axesFloat[1]/scale_factor_ellipse))

    """ Rectangle corners: not working for now

    top_left = (int((ellipseCenter[0] - axes[0]/2)/scale_factor_ellipse), int((ellipseCenter[1] - np.sin(-np.deg2rad(angle))*axes[0]/2)/scale_factor_ellipse))
    bottom_right = (int((ellipseCenter[0] + axes[0]/2)/scale_factor_ellipse), int((ellipseCenter[1] + axes[1]/2)/scale_factor_ellipse))
    cv2.circle(img, top_left, 5, (0, 255, 0), -1)
    cv2.putText(img, "top left", top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(img, bottom_right, 5, (0, 255, 0), -1)
    cv2.putText(img, "bottom right", bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    """
    rotatedRect = cv2.RotatedRect(resizedEllipse.center, resizedEllipse.axesFloat, resizedEllipse.angle)
    vertices = cv2.boxPoints(rotatedRect)
    vertices = np.int0(vertices)
    for i in range(4):
        cv2.line(img, tuple(vertices[i]), tuple(vertices[(i+1)%4]), (255, 255, 0), 2)
    #brect = cv2.boundingRect(vertices)
    #cv2.rectangle(img, (brect[0], brect[1]), (brect[0]+brect[2], brect[1]+brect[3]), (255, 255, 0), 10) # draw the rotated rectangle
    #cv2.circle(img, (int(ellipseCenter[0]/scale_factor_ellipse), int(ellipseCenter[1]/scale_factor_ellipse)), 10, (0, 255, 0), -1) #draw the center
    cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 5) #draw the bounding box

    """ Draw ellipse and points on it: ----------------------------------------- """
    
    cv2.ellipse(img, (resizedEllipse.center, resizedEllipse.axesFloat, resizedEllipse.angle), (255, 0, 255), 2) #draw the ellipse

    ellipsePoints = get_ellipse_axis_intersections(resizedEllipse.center, resizedEllipse.axesFloat, resizedEllipse.angle)
    ellipseTop, elliseBottom, ellipseLeft, ellipseRight = ellipsePoints
    cv2.circle(img, (int(ellipseTop[0]), int(ellipseTop[1])), 2, (0, 255, 0), -1)
    #cv2.putText(img, "EllipseTop", (int(ellipseTop[0]), int(ellipseTop[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (int(elliseBottom[0]), int(elliseBottom[1])), 2, (0, 255, 0), -1)
    #cv2.putText(img, "EllipseBottom", (int(elliseBottom[0]), int(elliseBottom[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (int(ellipseLeft[0]), int(ellipseLeft[1])), 2, (0, 255, 0), -1)
    #cv2.putText(img, "EllipseLeft", (int(ellipseLeft[0]), int(ellipseLeft[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (int(ellipseRight[0]), int(ellipseRight[1])), 2, (0, 255, 0), -1)
    #cv2.putText(img, "EllipseRight", (int(ellipseRight[0]), int(ellipseRight[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)

    """ demo_resize = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    cv2.imshow("Bounding box preview:", demo_resize)
    cv2.waitKey() """

    """ draw the inferred circle: """
    #cv2.circle(img, remapped_board_center_color, 2, (255, 0, 255), -1)
    #cv2.circle(img, remapped_board_center_color, inferred_radius, (100, 100, 0), 5)

    """ writing image and text files: """
    """     cv2.imwrite(osp.join("boards/board_angles/annotated", "cropped_"+img_name), img)
    print("wrote img: boards/board_angles/annotated/cropped_" + img_name)

    with open(osp.join('boards/board_angles/annotated', "cropped_" + osp.splitext(img_name)[0] + '.txt'), "w") as f:
        f.write("Ellipse: center=" + str(rescaled_ellipseCenter_int) + " axes=" + str(rescaled_axes) + " angle=" + str(angle) + "\n")
        f.write("Board center color=" + str(remapped_board_center_color) + "\n")
        f.write("Board center Hough: to be implemented" + "\n")
        f.write("Center difference: " + str(np.array(remapped_board_center_color) - np.array(rescaled_ellipseCenter_int)) + "\n")
        f.write("Bounding box: " + str(reformatted_bbox) + "\n")
        f.write("Ellipse Points: " + str(ellipsePoints) + "\n")
      """

    return reformatted_bbox, remapped_redEllipse

def resize_ellipse(scale_factor, ellipse):
    resizedEllipse = Ellipse()
    resizedEllipse.center = (int(ellipse.centerFloat[0]/scale_factor), int(ellipse.centerFloat[1]/scale_factor))
    resizedEllipse.centerFloat = (ellipse.centerFloat[0]/scale_factor, ellipse.centerFloat[1]/scale_factor)
    resizedEllipse.axes = (int(ellipse.axesFloat[0]/scale_factor), int(ellipse.axesFloat[1]/scale_factor))
    resizedEllipse.axesFloat = (ellipse.axesFloat[0]/scale_factor, ellipse.axesFloat[1]/scale_factor)
    resizedEllipse.angle = ellipse.angle
    return resizedEllipse

def compute_cal_pts(preciseEllipse, img): # compute cal pts

    cal_pts = None
    h, w = img.shape[:2]
    angle = preciseEllipse.angle

    full_BoardBBox = getSquareBboxForEllipse(preciseEllipse.centerFloat, preciseEllipse.axesFloat, h, w, 0.32)
    bbox_length = (full_BoardBBox[1][0] - full_BoardBBox[0][0]) # the 451mm diameter of the board 

    crop_imgToBBox = cv2.getRectSubPix(img, (bbox_length, bbox_length), preciseEllipse.centerFloat)
    top_crop = crop_imgToBBox[0:int(bbox_length*0.5), 0:bbox_length]
    lower_crop = crop_imgToBBox[int(bbox_length*0.5):bbox_length, 0:bbox_length]
    
    
    print("ellipse angle = ", angle)

    # angle zones if there is no rotation
    vertical_zone = (180-17, 180) # on the board the line would have angle 171° because hough lines are in [0, 180) degrees
    horizontal_zone = (90-17, 90) # on the board the line would have angle 81°

    #if there is rotation adjust the zones
    if(angle >= 0 and angle < 30): # ellipse rotated slightly to the right (at most 30°)
        angle = angle/10 # only slightly influence the angle because ellipse angle might be wrong (happens when image is taken from the front but ellipse is rotated instead of 0)
        horizontal_zone = (90+angle-17, 90+angle)
        vertical_zone = (180+angle-17, 180+angle)
    elif(angle > 150 and angle < 180): # ellipse rotated slightly to the left (at most 30°)
        angle = (180-angle)/10
        horizontal_zone = (90-angle-17, 90-angle)
        vertical_zone = (180-angle-17, 180-angle)

    vertical_line, horizontal_line = find_cal_sector_lines(crop_imgToBBox, vertical_zone, horizontal_zone, top=True)

    remapped_vertical_line = (remap_point(preciseEllipse, bbox_length, vertical_line[0]), remap_point(preciseEllipse, bbox_length, vertical_line[1]))
    remapped_horizontal_line = (remap_point(preciseEllipse, bbox_length, horizontal_line[0]), remap_point(preciseEllipse, bbox_length, horizontal_line[1]))

    """ top and lower seperate

    top_vertical, top_horizontal = find_cal_sector_lines(top_crop, vertical_zone, horizontal_zone, top=True)

    lower_vertical, lower_horizontal = find_cal_sector_lines(lower_crop, vertical_zone, horizontal_zone, top=False)
    
    cal1 = line_ellipse_intersection(preciseEllipse.centerFloat, preciseEllipse.axesFloat, preciseEllipse.angle, top_vertical[1]) # top crop
    cal4 = line_ellipse_intersection(preciseEllipse.centerFloat, preciseEllipse.axesFloat, preciseEllipse.angle, top_horizontal[1]) # top crop
    cal2 = line_ellipse_intersection(preciseEllipse.centerFloat, preciseEllipse.axesFloat, preciseEllipse.angle, lower_vertical[1]) # lower crop
    cal3 = line_ellipse_intersection(preciseEllipse.centerFloat, preciseEllipse.axesFloat, preciseEllipse.angle, lower_horizontal[1]) # lower crop
    """

    cal1_2 = line_ellipse_intersection(preciseEllipse.centerFloat, preciseEllipse.axesFloat, preciseEllipse.angle, remapped_vertical_line)
    cal3_4 = line_ellipse_intersection(preciseEllipse.centerFloat, preciseEllipse.axesFloat, preciseEllipse.angle, remapped_horizontal_line)

    cal1 = min(cal1_2, key=lambda y: y[1])
    cal2 = max(cal1_2, key=lambda y: y[1])
    cal3 = min(cal3_4, key=lambda x: x[0])
    cal4 = max(cal3_4, key=lambda x: x[0])

    cal_pts = [cal1, cal2, cal3, cal4]

    return cal_pts

def remap_point(preciseEllipse, bbox_length, point):
    offset_x = int(preciseEllipse.centerFloat[0] - bbox_length / 2)
    offset_y = int(preciseEllipse.centerFloat[1] - bbox_length / 2)
    return (point[0] + offset_x, point[1] + offset_y)

def find_furthest_intersections(mask, 
                              center,
                              axes,
                              num_angles= 360):
    """
    Find furthest intersection points from center at different angles.
    
    Args:
        mask: Binary image where arcs are white (255) on black background (0)
        center: (x, y) coordinate tuple of the ellipse center as floats
        num_angles: Number of angles to sample around the center
        
    Returns:
        Array of points shaped (N, 1, 2) ready for cv2.fitEllipse
    """
    height, width = mask.shape
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) # remove noise
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) # smooth sections
    closed = otsu_thresholding(closed) # threshold again to get rid of noise
 
    # possible radii
    max_radius = int(np.max(axes)/2*1.1) # 10 percent as buffer
    min_radius = int(np.min(axes)/2*0.9) # minimum 90 percent of the minor axis

    intersection_points = []
    
    # Sample angles
    for angle in np.linspace(0, 360, num_angles, endpoint=False):
        theta = np.deg2rad(angle)
        
        # Create line mask from center to edge at this angle
        end_x = int(center[0] + max_radius * np.cos(theta))
        end_y = int(center[1] + max_radius * np.sin(theta))
        
        line_mask = np.zeros_like(closed)
        cv2.line(line_mask, (int(center[0]), int(center[1])), (end_x, end_y), 255, 1)

        # Find intersections with arc segments
        intersection = cv2.bitwise_and(closed, line_mask)
        y_ints, x_ints = np.nonzero(intersection)
        
        if len(x_ints) > 0:
            # Calculate distances from center to intersection points
            distances = np.sqrt((x_ints - center[0])**2 + (y_ints - center[1])**2)
            # Get furthest point
            max_idx = np.argmax(distances)
            if(distances[max_idx] > min_radius):
                intersection_points.append([[float(x_ints[max_idx]), float(y_ints[max_idx])]])
    
    return np.array(intersection_points, dtype=np.float32)

def fit_ellipse_to_points(points, 
                         center = None,
                         known_axes= None):
    """
    Fit ellipse to points, optionally using known center and/or axes.
    
    Args:
        points: Array of points shaped (N, 1, 2)
        center: Optional (x, y) tuple to force ellipse center
        known_axes: Optional (width, height) tuple to force ellipse axes
        
    Returns:
        OpenCV ellipse parameters ((x,y), (width,height), angle)
    """
    returnEllipse = Ellipse()

    if len(points) < 5:
        raise ValueError("Need at least 5 points to fit ellipse")
    
    # Basic ellipse fit
    ellipse = cv2.fitEllipse(points)
    
    # If center is known, adjust the fit
    if center is not None:
        ellipse = (center, ellipse[1], ellipse[2])
    
    # If axes are known, adjust the fit
    if known_axes is not None:
        ellipse = (ellipse[0], known_axes, ellipse[2])
    
    returnEllipse.center = (int(ellipse[0][0]), int(ellipse[0][1]))
    returnEllipse.axes = (int(ellipse[1][0]), int(ellipse[1][1]))
    returnEllipse.centerFloat = ellipse[0]
    returnEllipse.axesFloat = ellipse[1]
    returnEllipse.angle = ellipse[2]
    return returnEllipse

def fit_ellipse_ransac(points, image_size, iterations=100, min_inliers=10):
    """
    Fit ellipse to points using RANSAC to handle outliers.
    
    Args:
        points: Array of points shaped (N, 1, 2)
        image_size: Tuple of (width, height) of the image
        iterations: Number of RANSAC iterations
        min_inliers: Minimum number of inliers required for a valid fit
        
    Returns:
        Best fitting Ellipse object or None if no valid fit found
    """
    if len(points) < 5:
        raise ValueError("Need at least 5 points to fit ellipse")
    
    # Set threshold as fraction of image size
    threshold = max(image_size) * 0.001
    
    best_ellipse = None
    max_inliers = 0
    sample_size = max(len(points) // 4, 6)
    points = points.reshape(-1, 2)  # Reshape to (N, 2) for easier handling
    
    def point_to_ellipse_distance(point, ellipse):
        """
        Calculate exact distance from point to ellipse using the ellipse equation.
        Returns the absolute difference between 1 and the evaluation of the point
        in the normalized ellipse equation.
        """
        center = ellipse[0]
        axes = ellipse[1]
        angle = ellipse[2]
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Translate point to origin
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        
        # Rotate point to align with ellipse axes
        x_rot = dx * cos_angle + dy * sin_angle
        y_rot = -dx * sin_angle + dy * cos_angle
        
        # Get semi-major and semi-minor axes
        a = axes[0] / 2
        b = axes[1] / 2
        
        # Evaluate ellipse equation: (x/a)^2 + (y/b)^2 = 1
        # Distance will be |1 - result|
        result = (x_rot/a)**2 + (y_rot/b)**2
        
        # Convert to pixel distance (approximate)
        pixel_distance = abs(1 - np.sqrt(result)) * np.sqrt((a**2 + b**2)/2)
        return pixel_distance
    
    all_indices = set(range(len(points)))
    
    for _ in range(iterations):
        try:
            # Randomly select 5 points (minimum needed for ellipse fitting)
            sample_indices = set(sample(range(len(points)), sample_size))
            sample_points = points[list(sample_indices)].reshape(-1, 1, 2)
            
            # Fit ellipse to sample
            candidate = cv2.fitEllipse(sample_points)
            
            # Count inliers
            non_sample_indices = all_indices - sample_indices
            inlier_indices = set(i for i in non_sample_indices 
                               if point_to_ellipse_distance(points[i], candidate) < threshold)
            inliers = len(inlier_indices) + sample_size
            # Update best solution if we found more inliers
            if inliers > max_inliers and inliers >= min_inliers:
                max_inliers = inliers
                best_ellipse = candidate
                if (inliers == 0.9*len(points)): # terminate early
                    break
                
        except cv2.error:
            # Skip if ellipse fitting fails for this sample
            continue
    
    if best_ellipse is None:
        return None
        
    # Create return object
    returnEllipse = Ellipse()
    returnEllipse.center = (int(best_ellipse[0][0]), int(best_ellipse[0][1]))
    returnEllipse.axes = (int(best_ellipse[1][0]), int(best_ellipse[1][1]))
    returnEllipse.centerFloat = best_ellipse[0]
    returnEllipse.axesFloat = best_ellipse[1]
    returnEllipse.angle = best_ellipse[2]
    
    return returnEllipse

def rotate_points(points: np.ndarray, angle_deg: float, center): # find ellipse line intersections pt 2
    """Rotate points around a center point by given angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Translate points to origin
    translated = points - np.array(center)
    
    # Rotation matrix
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                              [sin_theta, cos_theta]])
    
    # Rotate and translate back
    rotated = np.dot(translated, rotation_matrix.T)
    return rotated + np.array(center)

def line_ellipse_intersection(center, axes, angle, line_points):  # find ellipse line intersections pt 2
    """
    Find intersection points between an ellipse and a line.
    
    Args:
        ellipse: OpenCV ellipse format ((x,y), (width,height), angle)
        line_points: Two points defining the line ((x1,y1), (x2,y2))
    
    Returns:
        List of intersection points (can be empty, 1 point, or 2 points)
    """
    # Extract ellipse parameters
    (cx, cy), (b, a), angle = center, axes, angle # Note: b is width, a is height
    
    # Convert line points to numpy arrays
    p1 = np.array(line_points[0])
    p2 = np.array(line_points[1])
    
    # Step 1: Translate ellipse center to origin
    p1 = p1 - np.array([cx, cy])
    p2 = p2 - np.array([cx, cy])
    
    # Step 2: Rotate points so ellipse is axis-aligned
    points = np.vstack([p1, p2])
    rotated_points = rotate_points(points, -angle, (0, 0))
    p1, p2 = rotated_points
    
    # Step 3: Scale to unit circle (x² + y² = 1)
    p1 = p1 / np.array([b/2, a/2])
    p2 = p2 / np.array([b/2, a/2])
    
    # Step 4: Solve quadratic equation for intersection with unit circle
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    A = dx*dx + dy*dy
    B = 2 * (p1[0]*dx + p1[1]*dy)
    C = p1[0]*p1[0] + p1[1]*p1[1] - 1
    
    discriminant = B*B - 4*A*C
    
    if discriminant < 0:
        return []  # No intersection
    
    # Find intersection parameters
    t1 = (-B + np.sqrt(discriminant)) / (2*A)
    t2 = (-B - np.sqrt(discriminant)) / (2*A)
    
    intersections = []
    
    # Step 5: Convert back to original ellipse
    for t in [t1, t2]:
        if 0 <= t <= 1:  # Check if intersection is on line segment
            # Intersection with unit circle
            x = p1[0] + t*dx
            y = p1[1] + t*dy
            
            # Scale back
            x = x * (b/2)
            y = y * (a/2)
            
            # Rotate back
            point = rotate_points(np.array([[x, y]]), angle, (0, 0))[0]
            
            # Translate back
            point = point + np.array([cx, cy])
            
            intersections.append(tuple(point))
    
    return intersections

def ellipse2circle(center, axes, angle):  # find ellipse line intersections pt 1 failes
    angleRad = np.deg2rad(angle)
    x = center[0]
    y = center[1]
    b = axes[0]
    a = axes[1]

    # build transformation matrix http://math.stackexchange.com/questions/619037/circle-affine-transformation
    R1 = np.array([[np.cos(angleRad), np.sin(angleRad), 0], [-np.sin(angleRad), np.cos(angleRad), 0], [0, 0, 1]])
    R2 = np.array([[np.cos(angleRad), -np.sin(angleRad), 0], [np.sin(angleRad), np.cos(angleRad), 0], [0, 0, 1]])

    T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    T2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    D = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])

    M = T2.dot(R2.dot(D.dot(R1.dot(T1))))

    return M

def getEllipseLineIntersection(center, axes, M, lines_seg):  # find ellipse line intersections pt 1
    circle_radius = axes[1] # minor axis as radius?
    M_inv = np.linalg.inv(M)

    # find line circle intersection and use inverse transformation matrix to transform it back to the ellipse
    intersectp_s = []
    for lin in lines_seg:
        line_p1 = M.dot(np.transpose(np.hstack([lin[0], 1])))
        line_p2 = M.dot(np.transpose(np.hstack([lin[1], 1])))
        inter1, inter_p1, inter2, inter_p2 = intersectLineCircle(np.asarray(center), circle_radius,
                                                                 np.asarray(line_p1), np.asarray(line_p2))
        if inter1:
            inter_p1 = M_inv.dot(np.transpose(np.hstack([inter_p1, 1])))
            if inter2:
                inter_p2 = M_inv.dot(np.transpose(np.hstack([inter_p2, 1])))
                intersectp_s.append(inter_p1)
                intersectp_s.append(inter_p2)

    return intersectp_s

def intersectLineCircle(center, radius, p1, p2):  # find ellipse line intersections pt 1
    baX = p2[0] - p1[0]
    baY = p2[1] - p1[1]
    caX = center[0] - p1[0]
    caY = center[1] - p1[1]

    a = baX * baX + baY * baY
    bBy2 = baX * caX + baY * caY
    c = caX * caX + caY * caY - radius * radius

    pBy2 = bBy2 / a
    q = c / a

    disc = pBy2 * pBy2 - q
    if disc < 0:
        return False, None, False, None

    tmpSqrt = np.sqrt(disc)
    abScalingFactor1 = -pBy2 + tmpSqrt
    abScalingFactor2 = -pBy2 - tmpSqrt

    pint1 = p1[0] - baX * abScalingFactor1, p1[1] - baY * abScalingFactor1
    if disc == 0:
        return True, pint1, False, None

    pint2 = p1[0] - baX * abScalingFactor2, p1[1] - baY * abScalingFactor2
    return True, pint1, True, pint2

def get_ellipse_axis_intersections(center, size, angle_deg):

    angle_rad = np.deg2rad(angle_deg) # we want to "undo" the rotation of the ellipse => rotate counterclockwise
    # Create rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])
    
    # Get semi-axes lengths
    a, b = size[0] / 2, size[1] / 2
    
    # Points along major and minor axes before rotation
    axis_points_local = np.array([[0, -b], [0, b], [-a, 0], [a, 0]]) # top, bottom, left, right
    
    # Rotate points and add center offset
    axis_points = []
    for point in axis_points_local:
        moved = (R @ point.T).T + center
        axis_points.append(moved)
    
    return axis_points

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def find_cal_sector_lines(img, vertical_zone, horizontal_zone, top=True): # returns vertical and horizontal lines in polar and cartesian
    
    img_size = max(img.shape)
    #print("angle = ", angle)
    #print("img size = ", img_size)

    kernel = np.ones((5, 5), np.float32) / 25
    #blur = cv2.filter2D(img, -1, kernel) dont blur image for houghlines

    h, s, value = cv2.split(img)
    #cv2.imshow("value", value)

    ret, thresh = cv2.threshold(value, 140, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("thresh", thresh)

    # removes border wire outside the outerellipse
    kernel = np.ones((5, 5), np.uint8)
    thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)

    #cv2.imshow("thresh2", thresh2)
    edged = cv2.Canny(thresh2, 250, 255)
    autocanny = auto_canny(thresh2)

    """ cv2.imshow("autocanny", autocanny)
    cv2.waitKey()
    cv2.destroyAllWindows() """

    p = []
    intersectp = []
    lines_seg = []
    counter = 0

    if vertical_zone is None:
        vertical_zone = (180-17, 180), (360-17, 360)
    if horizontal_zone is None:
        horizontal_zone = (90-17, 90), (270-17, 270)

    # Fit line to find intersection point for dartboard center point
    lines = cv2.HoughLines(autocanny, 1, np.pi/180, 50) # from extracted method in calibration_1.py
    vertical_candidate = None
    horizontal_candidate = None

    for line in lines: # find the first line that fits the vertical zone
        rho, theta = line[0]
        if np.deg2rad(vertical_zone[0]) < theta < np.deg2rad(vertical_zone[1]):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 4000*(-b))
            y1 = int(y0 + 4000*(a))
            x2 = int(x0 - 4000*(-b))
            y2 = int(y0 - 4000*(a))
            print("vertical angle: ", np.rad2deg(theta))
            vertical_candidate = [(x1, y1), (x2, y2)]
            break
    
    for line in lines: # find the first line that fits the horizontal zone
        rho, theta = line[0]
        if np.deg2rad(horizontal_zone[0]) < theta < np.deg2rad(horizontal_zone[1]):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 4000*(-b))
            y1 = int(y0 + 4000*(a))
            x2 = int(x0 - 4000*(-b))
            y2 = int(y0 - 4000*(a))
            print("horizontal angle: ", np.rad2deg(theta))
            horizontal_candidate = [(x1, y1), (x2, y2)]
            break

    return vertical_candidate, horizontal_candidate        

    strongest_10_lines = [lines[0][0]]

    for line in lines:
        differs = True
        rho, theta = line[0]
        if len(strongest_10_lines) < 10:
            for strong_line in strongest_10_lines:
                if abs(strong_line[1] - theta) <= np.deg2rad(9): # at least 9 degrees difference
                    differs = False
                    break
            if differs:
                strongest_10_lines.append(line[0])
        else:
            break
    #print("strongest 10 lines = ", len(strongest_10_lines))

    """ from calibration.py
    anglezone1 = (angle - 5, angle + 5) 
    anglezone2 = (angle - 100, angle - 80)
    lines2 = cv2.HoughLines(edged, 1, np.pi / 70, 100, 100)
    """


    """draw lines over whole image: """
    for line in strongest_10_lines:
            rho,theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 4000*(-b))
            y1 = int(y0 + 4000*(a))
            x2 = int(x0 - 4000*(-b))
            y2 = int(y0 - 4000*(a))
            lines_seg.append([(x1, y1), (x2, y2)])
    


    """ (for demo) differentiate top and bottom: """
    """ if top:
        for line in strongest_10_lines:
            rho,theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            if(theta <= np.pi/2): # 1 quadrant => a >= 0, b > 0
                x1 = int(x0 + 4000*(-b))
                y1 = int(y0 + 4000*(a))
                x2 = int(x0 - 4000*(-b))
                y2 = int(y0 - 4000*(a))
                lines_seg.append([(x1, y1), (x2, y2)])
                #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
            if(theta > np.pi/2): # 4 quadrant => a < 0, b > 0
                x1 = int(x0 + 4000*(-b))
                y1 = int(y0 + 4000*(a))
                x2 = int(x0 - 4000*(-b))
                y2 = int(y0 - 4000*(a))
                lines_seg.append([(x1, y1), (x2, y2)])
                #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    else:
        for line in strongest_10_lines:
            rho,theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 4000*(-b))
            y1 = int(y0 + 4000*(a))
            x2 = int(x0 - 4000*(-b))
            y2 = int(y0 - 4000*(a))
            lines_seg.append([(x1, y1), (x2, y2)])
            #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2) """


    #cv2.imwrite(osp.join("hough/", "Hough_"+img_name), img)

    return lines_seg, strongest_10_lines

    print("vertical[0] = ", np.deg2rad(vertical_zone[0]))
    print("vertical[1] = ", np.deg2rad(vertical_zone[1]))
    if lines is None:
        return lines_seg

    # Sector angles important -> make accessible
    for rho_theta in sorted_lines:
        rho, theta = rho_theta[0]
        print("theta in deg: = ", np.rad2deg(theta))
        print("rho = ", rho)

        # Split between horizontal and vertical lines (take only lines in certain range)
        if np.deg2rad(vertical_zone[0][0]) < theta < np.deg2rad(vertical_zone[0][1]) or np.deg2rad(vertical_zone[1][0]) < theta < np.deg2rad(vertical_zone[1][1]):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * a)
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * a)
            print("x0 = ", x0, "y0 = ", y0) 
            cv2.circle(img, (int(x0), int(y0)), 10, (0, 0, 255), -1)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.imshow("lines", img)
            cv2.waitKey()
            continue

            for rho1_theta1 in lines:
                rho1, theta1 = rho1_theta1[0]
                print("theta1 = ", theta1)

                if np.pi / 180 * angle_zone2[0] < theta1 < np.pi / 180 * angle_zone2[1]:
                    a = np.cos(theta1)
                    b = np.sin(theta1)
                    x0 = a * rho1
                    y0 = b * rho1
                    x3 = int(x0 + 2000 * (-b))
                    y3 = int(y0 + 2000 * a)
                    x4 = int(x0 - 2000 * (-b))
                    y4 = int(y0 - 2000 * a)

                    if y1 == y2 and y3 == y4:  # Horizontal Lines
                        diff = abs(y1 - y3)
                    elif x1 == x2 and x3 == x4:  # Vertical Lines
                        diff = abs(x1 - x3)
                    else:
                        diff = 0

                    if diff < 200 and diff != 0:
                        continue

                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                    p.append((x1, y1))
                    p.append((x2, y2))
                    p.append((x3, y3))
                    p.append((x4, y4))

                    intersectpx, intersectpy = intersect_lines(
                        p[counter], p[counter + 1], p[counter + 2], p[counter + 3]
                    )

                    # Consider only intersections close to the center of the image
                    if intersectpx < 200 or intersectpx > 900 or intersectpy < 200 or intersectpy > 900:
                        continue

                    intersectp.append((intersectpx, intersectpy))

                    lines_seg.append([(x1, y1), (x2, y2)])
                    lines_seg.append([(x3, y3), (x4, y4)])

                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 1)

                    # Point offset
                    counter += 4

    return lines_seg, img

def intersect_lines(p1, p2, p3, p4):
    """
    Calculate the intersection point of two lines defined by points p1, p2 and p3, p4.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Line equations
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return float('inf'), float('inf')  # Parallel lines

    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return int(intersect_x), int(intersect_y)

def rotate_image(image, rotation_center, angle):
    image_center = rotation_center
    if rotation_center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2) 
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

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

def workingfindEllipse(img, original_img=None, scale_factor=1):
    """
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
    print("img shape = ", img.shape)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #closed = otsu_thresholding(closed)

    """ cv2.imshow("closed", closed)
    cv2.waitKey() """


    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("found contours = ", len(contours))
    contoured = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    """ cv2.imshow("contours", contoured)
    cv2.waitKey()
    cv2.destroyAllWindows() """


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
    print("size of largest ellipse = ", size_of_largest_ellipse)

    for i, ellipse in enumerate(ellipses):
        if ellipse[2] > 0.8:
            print("ellipse i size: ", ellipse[1])
            if ellipse[1] > 0.5 * size_of_largest_ellipse:
                best_ellipse = ellipse[0] # for now its just the largest ellipse with circularity > 0.8 and it must be at least 50% of the largest ellipse
            break
    if best_ellipse is None:
        best_ellipse = ellipses[0][0] # if no ellipse with circularity > 0.8, return the largest ellipse
    return best_ellipse

def findEllipseInRed(img, img_name=None, original_img=None, scale_factor=1):
    """
    input: binary image cropped to board
    output: params of roated rectangle around ellipse
    center point, (width, height)=axes=(b,a)=(minor,major), angle
    """  
    img = cv2.GaussianBlur(img.copy(), (3, 3), 0)
    print("img shape = ", img.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #closed = otsu_thresholding(closed)

    cv2.imshow("closed", closed)
    cv2.waitKey()


    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("found contours = ", len(contours))
    contoured = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow("contours", contoured)
    cv2.waitKey()
    cv2.destroyAllWindows()


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
    print("size of largest ellipse = ", size_of_largest_ellipse)

    for i, ellipse in enumerate(ellipses):
        if ellipse[2] > 0.8:
            print("ellipse i size: ", ellipse[1])
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
    print(axes)
    scale_factor = 1 + random.uniform(0.3, 0.35) # outer rim is 32% larger than the ellipse
    if buffer is not None:
        scale_factor = 1 + buffer
    long_axis = max(axes) * scale_factor # set the bounding box to be 25-35% larger than the ellipse
    print(long_axis)
    top_left_corner = (max(0, int(center[0] - long_axis/2)), 
                      max(0, int(center[1] - long_axis/2)))
    bottom_right_corner = (min(w, int(center[0] + long_axis/2)), 
                          min(h, int(center[1] + long_axis/2)))
    
    return (top_left_corner, bottom_right_corner)

def find_boardCenterMinEnclosing(crop_img):

    img_opened = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cv2.imshow("opened", img_opened)
    cv2.waitKey()
    crop_img_closed_circle = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cv2.imshow("closed", crop_img_closed_circle)
    cv2.waitKey()
    contours, hierarchy = cv2.findContours(img_opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours = ", len(contours))
    nr_of_levels = len(hierarchy[0])
    print("nr of levels = ", nr_of_levels)
    best_circle = cv2.minEnclosingCircle(contours[len(contours)-1])
    """ for i, c in enumerate(contours):
        # take contour with no child
        if hierarchy[0][i][2] == -1:
            best_cirlce = cv2.minEnclosingCircle(c)
        if c.shape[0] > 5:
            (x, y), radius = cv2.minEnclosingCircle(c)
            if best_cirlce is None or radius < best_cirlce[1]:
                best_cirlce = ((x, y), radius)
 """
    cv2.circle(img_opened, (int(best_circle[0][0]), int(best_circle[0][1])), int(best_circle[1]), (0, 255, 0), 10)
    #cv2.imshow("circle", img_opened)
    cv2.waitKey()

    return contours, best_circle[0], int(best_circle[1]) 

def find_boardCenterHough(crop_img):
    cv2.imshow("Houghcircles", crop_img)
    circles = cv2.HoughCircles(crop_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    print("Found circles = ", circles)
    if circles is None:
        return None, None, None
    best_circle = circles[0][0]
    circles = np.uint16(np.around(circles))
    """   for i in circles[0, :]:
        cv2.circle(crop_img, (i[0], i[1]), i[2], (0, 255, 0), 10)
        cv2.circle(crop_img, (i[0], i[1]), 2, (0, 0, 255), 3) """
    cv2.circle(crop_img, (int(best_circle[0]), int(best_circle[1])), int(best_circle[2]), (0, 255, 0), 10)
    cv2.imshow("Houghcircles", crop_img)
    cv2.waitKey()
    return best_circle


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Board detection using ellipse fitting.')
    parser.add_argument('-i', '--input', help='Path to input image.', 
                       default='boards/myboard/testboard.jpg')
    args = parser.parse_args()
    
    if osp.isdir(args.input):
        for img in os.listdir(args.input):
            if not osp.isdir(osp.join(args.input, img)):
                if (img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')):
                    print("bbox = ", find_board_vEllipse2(osp.join(args.input, img)))
                else:
                    print("Skipping non image file: ", img)
    else:
        print("bbox = ", find_board_vEllipse2(args.input))