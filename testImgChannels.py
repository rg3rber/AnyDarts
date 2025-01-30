from PIL import Image
import numpy as np
import cv2

# Load image with PIL
img_pil = Image.open("dataset/holo_v1/train/images/02_12_2024_IMG_0.jpg")
img_cv2 = cv2.imread("dataset/holo_v1/train/images/02_12_2024_IMG_0.jpg")

# Check mode
if img_pil.mode == "RGB":
    print("The image is originally RGB")
else:
    print(f"The image mode is {img_pil.mode}")

# Swap R and B channels
swapped = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

# Check if the swap significantly changes the image
if np.array_equal(img_cv2, swapped):
    print("The image is likely RGB (unchanged by swap)")
else:
    print("The image is likely BGR (swap changed colors)")
