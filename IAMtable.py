import cv2
import numpy as np

# Load 4 random word images from your IAM dataset folder
img1 = cv2.imread("E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/c06-043-00-00.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/c06-043-00-01.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/c06-043-00-02.png", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/c06-043-00-03.png", cv2.IMREAD_GRAYSCALE)

# Resize them to a uniform block size so they fit in a grid nicely (e.g., 200x100)
dim = (200, 100)
img1 = cv2.resize(img1, dim)
img2 = cv2.resize(img2, dim)
img3 = cv2.resize(img3, dim)
img4 = cv2.resize(img4, dim)

# Add a black border around each to simulate table lines
border = 2
border_color = (0, 0, 0) # Tuple format fixes the Pylance warning

img1 = cv2.copyMakeBorder(img1, border, border, border, border, cv2.BORDER_CONSTANT, value=border_color)
img2 = cv2.copyMakeBorder(img2, border, border, border, border, cv2.BORDER_CONSTANT, value=border_color)
img3 = cv2.copyMakeBorder(img3, border, border, border, border, cv2.BORDER_CONSTANT, value=border_color)
img4 = cv2.copyMakeBorder(img4, border, border, border, border, cv2.BORDER_CONSTANT, value=border_color)

# Concatenate horizontally to make rows
row1 = cv2.hconcat([img1, img2])
row2 = cv2.hconcat([img3, img4])

# Concatenate vertically to make the final table
table_img = cv2.vconcat([row1, row2])

# Save the synthetic table
cv2.imwrite('synthetic_iam_table.jpg', table_img)
print("Synthetic IAM table created successfully!")