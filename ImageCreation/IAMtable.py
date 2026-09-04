"""
This script is used to create a table of images words from IAM tables dataset
"""

import cv2
import numpy as np
import os
import glob
import math

# ---------- Folder containing the word images ----------
# Just point this at the folder and every image inside it will be used
image_folder ="E:/PROJECTS/APT_Summer_Project/TestImage/textImages/TextToTableImage(3)"

# Which extensions to pick up from the folder
valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

image_paths = sorted(
    p for p in glob.glob(os.path.join(image_folder, "*"))
    if p.lower().endswith(valid_exts)
)

if not image_paths:
    raise ValueError(f"No images found in folder: {image_folder}")

cols = 2
# rows are computed automatically from however many images were found in the folder
rows = math.ceil(len(image_paths) / cols)

# [FIX 1] Increased padding to give the YOLO model enough whitespace to differentiate text from borders
inner_padding = 25         
line_thickness = 3         # thickness of the table grid lines
row_bg_color = 230         # light gray for alternating rows
outer_margin = 50          # white margin around the whole table

# ---------- Load images (no extra border) ----------
images = []
for p in image_paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Image not found: {p}")
        continue

    # Only white inner padding – no black border around the word
    img = cv2.copyMakeBorder(img, inner_padding, inner_padding,
                             inner_padding, inner_padding,
                             borderType=cv2.BORDER_CONSTANT, value=255)
    images.append(img)

if not images:
    raise ValueError("No images were loaded. Check your folder path.")

# ---------- Determine column/row sizes ----------
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

# Inner cell size = padded word size (no extra border)
col_inner = [w for w in col_widths]
row_inner = [h for h in row_heights]

# Cell total size includes one grid line to the right / below
col_cell = [w + line_thickness for w in col_inner]
row_cell = [h + line_thickness for h in row_inner]

# ---------- Canvas ----------
table_w = sum(col_cell) + line_thickness + 2 * outer_margin
table_h = sum(row_cell) + line_thickness + 2 * outer_margin
table_img = np.ones((table_h, table_w), dtype=np.uint8) * 255

# Top‑left of each cell’s inner area (after grid line)
x_starts = []
x = outer_margin + line_thickness
for cw in col_cell:
    x_starts.append(x)
    x += cw

y_starts = []
y = outer_margin + line_thickness
for rh in row_cell:
    y_starts.append(y)
    y += rh

# ---------- 1. Alternating row background ----------
for r in range(rows):
    if r % 2 == 1:
        y_top = y_starts[r]
        y_bottom = y_top + row_inner[r]
        for c in range(cols):
            x_left = x_starts[c]
            x_right = x_left + col_inner[c]
            table_img[y_top:y_bottom, x_left:x_right] = row_bg_color

# ---------- 2. Place the padded word images (centered) ----------
# [FIX 2] Move this step BEFORE drawing grid lines, and use cv2.minimum to blend
for i, img in enumerate(images):
    r = i // cols
    c = i % cols
    h, w = img.shape

    x_offset = (col_inner[c] - w) // 2
    y_offset = (row_inner[r] - h) // 2
    x_pos = x_starts[c] + x_offset
    y_pos = y_starts[r] + y_offset

    roi = table_img[y_pos:y_pos+h, x_pos:x_pos+w]

    # By using minimum, the white background (255) of the image yields to the
    # gray background (230) of the row, while the dark text pixels are preserved.
    # This entirely eliminates the artificial "white box" effect.
    table_img[y_pos:y_pos+h, x_pos:x_pos+w] = cv2.min(roi, img)

# ---------- 3. Draw table grid lines (the actual cells) ----------
# [FIX 3] Draw lines LAST so the white padding of the images doesn't overwrite them
# Vertical lines
for c in range(cols + 1):
    if c == 0:
        x = outer_margin
    else:
        x = x_starts[c-1] + col_inner[c-1] + line_thickness
    cv2.line(table_img, (x, outer_margin), (x, table_h - outer_margin),
             color=0, thickness=line_thickness)

# Horizontal lines
for r in range(rows + 1):
    if r == 0:
        y = outer_margin
    else:
        y = y_starts[r-1] + row_inner[r-1] + line_thickness
    cv2.line(table_img, (outer_margin, y), (table_w - outer_margin, y),
             color=0, thickness=line_thickness)

# ---------- Save ----------
cv2.imwrite("synthetic_iam_table_cells_only_fixed.jpg", table_img)
print(f"Table with clean, continuous cells created from {len(images)} images "
      f"({rows} rows x {cols} cols)!")
