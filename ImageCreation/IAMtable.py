"""
This script is used to create a table of images words from IAM tables dataset
"""

import cv2
import numpy as np

# Word images (10 files)
image_paths = [
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-00-00.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-00-01.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-00-02.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-00-03.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-00-04.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-00-05.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-00-06.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-01-00.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-01-01.png",
    "E:/PROJECTS/OCRSoftware/TestImage/textImages/IAM images/table1/a01-000u-01-02.png",
]

rows = 5
cols = 2
padding = 8                    # inner margin around the word inside the cell
line_thickness = 3
row_bg_color = 230             # light gray for alternating rows

# Load all images
images = []
for p in image_paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {p}")
    images.append(img)

# Compute max width per column and max height per row
col_widths = [0] * cols
row_heights = [0] * rows
for i, img in enumerate(images):
    r = i // cols
    c = i % cols
    h, w = img.shape
    if w > col_widths[c]:
        col_widths[c] = w
    if h > row_heights[r]:
        row_heights[r] = h

# Add padding to get the inner cell dimensions
col_inner = [w + 2 * padding for w in col_widths]
row_inner = [h + 2 * padding for h in row_heights]

# Total cell dimensions (including the thickness of the grid lines)
col_cell = [w + line_thickness for w in col_inner]   # cell width = inner + one line
row_cell = [h + line_thickness for h in row_inner]   # cell height = inner + one line

# Table canvas size
table_w = sum(col_cell) + line_thickness   # extra line at right border
table_h = sum(row_cell) + line_thickness   # extra line at bottom border

# Create a white canvas
table_img = np.ones((table_h, table_w), dtype=np.uint8) * 255

# Pre-compute starting x/y of each cell's inner area (top-left of inner region)
x_starts = []
x = line_thickness
for cw in col_cell:
    x_starts.append(x)             # line left of cell
    x += cw
# y_starts similarly
y_starts = []
y = line_thickness
for rh in row_cell:
    y_starts.append(y)
    y += rh

# Draw alternating row backgrounds (inside the inner areas, before drawing text)
for r in range(rows):
    if r % 2 == 1:                 # shade every second row
        y_top = y_starts[r]
        y_bottom = y_top + row_inner[r]
        # For each column, fill the inner cell area
        for c in range(cols):
            x_left = x_starts[c]
            x_right = x_left + col_inner[c]
            table_img[y_top:y_bottom, x_left:x_right] = row_bg_color

# Draw the grid lines (vertical and horizontal)
# Vertical lines at every x_start and at the end of the table
for c in range(cols + 1):
    if c == 0:
        x = 0
    else:
        x = x_starts[c-1] + col_inner[c-1] + line_thickness  # right after inner + line
    cv2.line(table_img, (x, 0), (x, table_h), color=0, thickness=line_thickness)

# Horizontal lines
for r in range(rows + 1):
    if r == 0:
        y = 0
    else:
        y = y_starts[r-1] + row_inner[r-1] + line_thickness
    cv2.line(table_img, (0, y), (table_w, y), color=0, thickness=line_thickness)

# Place each word image centered inside its inner cell area
for i, img in enumerate(images):
    r = i // cols
    c = i % cols

    # The word is placed as is – no resizing (if you want uniform size, you could scale)
    h, w = img.shape

    # Center coordinates inside the inner cell
    x_offset = (col_inner[c] - w) // 2
    y_offset = (row_inner[r] - h) // 2
    x_pos = x_starts[c] + x_offset
    y_pos = y_starts[r] + y_offset

    # Put the word image onto the canvas
    table_img[y_pos:y_pos+h, x_pos:x_pos+w] = img

# Save the result
cv2.imwrite("synthetic_iam_table_tight_cells_3.jpg", table_img)
print("Table with word‑tight cell borders created successfully!")