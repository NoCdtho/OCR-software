import cv2
import numpy as np

def deskew_image(image):
    # 1. Convert to grayscale and invert (background black, text white)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # 2. Thresholding to create a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # 3. Get all white pixel coordinates containing text
    coords = np.column_stack(np.where(thresh > 0))
    
    # 4. Find the minimum area rectangle enclosing all text and extract the angle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle based on cv2.minAreaRect behavior
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # 5. Calculate the rotation matrix and apply affine transformation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the image with a white background border
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated