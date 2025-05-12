import cv2
import numpy as np

def extract_dartboard(image_path, output_path=None):
    """
    Extract the dartboard from the background and create a masked version.
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the result. If None, just returns the result
    
    Returns:
        tuple: (masked_color_image, mask)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not open image")
    
    # Create a window for trackbar
    cv2.namedWindow('Adjust Threshold')
    
    def on_threshold_change(x):
        # This will be called whenever the trackbar value changes
        pass
    
    # Create trackbar
    cv2.createTrackbar('Threshold', 'Adjust Threshold', 128, 255, on_threshold_change)
    
    # Initialize mask
    mask = None
    masked_img = None
    
    while True:
        # Get current threshold value
        thresh_val = cv2.getTrackbarPos('Threshold', 'Adjust Threshold')
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (should be the dartboard)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Apply mask to original image
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            # Show result
            cv2.imshow('Adjust Threshold', masked_img)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'enter' is pressed, keep the current mask
        if key == 13:  # Enter key
            break
        # If 'esc' is pressed, cancel
        elif key == 27:  # Esc key
            cv2.destroyAllWindows()
            return None, None
    
    cv2.destroyAllWindows()
    
    if output_path:
        # For PNG output with transparency
        # Convert mask to 4 channel (RGBA) where black becomes transparent
        rgba = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
        # Set alpha channel to 0 where mask is 0
        rgba[:, :, 3] = mask
        cv2.imwrite(output_path, rgba)
    
    return masked_img, mask

def refine_dartboard_section(image, mask, inner_ratio=0.8, outer_ratio=0.95):
    """
    Extract just the scoring section of the dartboard using ratios.
    
    Args:
        image: The masked dartboard image
        mask: The binary mask of the dartboard
        inner_ratio: Ratio to exclude inner section (numbers)
        outer_ratio: Ratio to exclude outer ring
    
    Returns:
        tuple: (refined_image, refined_mask)
    """
    # Find the center and radius using the mask
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        raise ValueError("No dartboard found in mask")
        
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    
    # Find the approximate radius using the mask area
    area = cv2.countNonZero(mask)
    radius = int(np.sqrt(area / np.pi))
    
    # Create new mask for the scoring section
    scoring_mask = np.zeros_like(mask)
    
    # Draw the outer and inner circles
    cv2.circle(scoring_mask, (center_x, center_y), int(radius * outer_ratio), 255, -1)
    cv2.circle(scoring_mask, (center_x, center_y), int(radius * inner_ratio), 0, -1)
    
    # Apply the new mask to the image
    refined_image = cv2.bitwise_and(image, image, mask=scoring_mask)
    
    return refined_image, scoring_mask

# Example usage:
if __name__ == "__main__":
    # First extract the dartboard from background
    masked_img, mask = extract_dartboard("images/dartboard.jpg")
    
    if masked_img is not None:
        # Then extract just the scoring section
        scoring_section, scoring_mask = refine_dartboard_section(masked_img, mask)
        
        # Save results
        cv2.imwrite("images/dartboard_masked.png", masked_img)
        cv2.imwrite("images/dartboard_scoring.png", scoring_section)