'''
1. This script extracts a structured table (as a pandas DataFrame) from an image in three stages:
Table Detection (YOLO) – Detects rows, columns, and spanning (merged) cells. It then builds a logical grid by 
computing row‑column intersections, while spanning cells occupy multiple grid positions. Each cell is assigned a 
(row, col) index and a bounding box.

2. Text Recognition (CRNN) – Crops the image at each unique bounding box, preprocesses the crops 
(grayscale, resize to fixed height), and feeds them in batches to a CTC‑based CRNN trained on handwriting (IAM). 
The recognized strings are mapped back to the boxes.

3. Grid Assembly – Places the recognized text into a 2‑D matrix using the pre‑computed row/column indices. Spanning cells get the same text in all covered positions (written only once to avoid duplication). The result is saved as a CSV.
'''


import cv2
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
from torchvision.transforms import functional as F
from CRNN.CRNN_model import CRNN   # your CRNN class


# 1. Configuration
YOLO_WEIGHTS = "E:/PROJECTS/OCR/Trained models/Yolomodel.pt"
CRNN_WEIGHTS = "E:/PROJECTS/OCR/Trained models/IAMCRNN.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CRNN parameters (adjust to your training)
TARGET_HEIGHT = 32
MAX_WIDTH = 256
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"-!?"
NUM_CLASSES = len(CHARSET) + 1   # +1 for blank


# 2. Table Detector (YOLO + grid logic)
class TableDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.class_names = {
            0: "table",
            1: "table row",
            2: "table column",
            3: "table spanning cell"
        }

    def get_cells_with_grid_positions(self, image_path, conf=0.25, iou=0.45):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model(img, conf=conf, iou=iou)
        rows, cols, spanning_cells = [], [], []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names.get(cls_id, "")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if class_name == "table row":
                    rows.append((x1, y1, x2, y2))
                elif class_name == "table column":
                    cols.append((x1, y1, x2, y2))
                elif class_name == "table spanning cell":
                    spanning_cells.append((x1, y1, x2, y2))

        rows.sort(key=lambda r: r[1])   # top to bottom
        cols.sort(key=lambda c: c[0])   # left to right

        num_rows, num_cols = len(rows), len(cols)

        # Grid type hint: each cell is None or a tuple of 4 ints
        grid: list[list[None | tuple[int,int,int,int]]] = [
            [None for _ in range(num_cols)] for _ in range(num_rows)
        ]
        is_spanning = [[False for _ in range(num_cols)] for _ in range(num_rows)]

        # Place spanning cells
        for sp_box in spanning_cells:
            sx1, sy1, sx2, sy2 = sp_box
            row_indices = [i for i, (_, ry1, _, ry2) in enumerate(rows)
                           if max(ry1, sy1) < min(ry2, sy2)]
            col_indices = [j for j, (cx1, _, cx2, _) in enumerate(cols)
                           if max(cx1, sx1) < min(cx2, sx2)]
            for i in row_indices:
                for j in col_indices:
                    grid[i][j] = sp_box
                    is_spanning[i][j] = True

        # Fill remaining cells with intersections
        for i, (rx1, ry1, rx2, ry2) in enumerate(rows):
            for j, (cx1, cy1, cx2, cy2) in enumerate(cols):
                if grid[i][j] is not None:
                    continue
                ix1 = max(rx1, cx1)
                iy1 = max(ry1, cy1)
                ix2 = min(rx2, cx2)
                iy2 = min(ry2, cy2)
                if ix1 < ix2 and iy1 < iy2:
                    grid[i][j] = (ix1, iy1, ix2, iy2)
                    is_spanning[i][j] = False

        # Build flat list
        cells = []
        for i in range(num_rows):
            for j in range(num_cols):
                box = grid[i][j]
                if box is not None:
                    cells.append({
                        'row': i,
                        'col': j,
                        'box': box,
                        'is_spanning': is_spanning[i][j]
                    })
        return cells

# 3. CRNN OCR loader and batch function

# Load model once (global)
# You must match the constructor arguments of your CRNN class exactly.
crnn_model = CRNN(num_classes=NUM_CLASSES)   # maybe add other params if needed
crnn_model.load_state_dict(torch.load(CRNN_WEIGHTS, map_location=DEVICE))
crnn_model.to(DEVICE)
crnn_model.eval()

def preprocess_crop(crop_img: np.ndarray) -> torch.Tensor:
    """ Convert BGR crop to grayscale tensor (1, 32, W_padded). """
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
    w, h = pil_img.size
    new_w = int(w * (TARGET_HEIGHT / h))
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH
    pil_img = pil_img.resize((new_w, TARGET_HEIGHT), Image.Resampling.BILINEAR)
    img_tensor = F.to_tensor(pil_img)
    return img_tensor

def batch_ocr(crops: list, batch_size=32):
    """ crops: list of BGR numpy arrays. Returns list of strings. """
    all_texts = [""] * len(crops)
    idx_to_char = {i: c for i, c in enumerate(CHARSET, start=0)}  # 0 = blank

    for start in range(0, len(crops), batch_size):
        batch = crops[start:start+batch_size]
        tensors = [preprocess_crop(c) for c in batch]
        max_w = max(t.shape[2] for t in tensors)

        padded = torch.zeros(len(tensors), 1, TARGET_HEIGHT, max_w)
        lengths = []
        for j, t in enumerate(tensors):
            _, h, w = t.shape
            padded[j, :, :h, :w] = t
            lengths.append(w)

        padded = padded.to(DEVICE)
        lengths = torch.tensor(lengths, dtype=torch.long)

        with torch.no_grad():
            log_probs = crnn_model(padded)   # shape (T, B, num_classes)

        _, max_indices = log_probs.max(dim=2)   # (T, B)
        max_indices = max_indices.permute(1, 0).cpu().numpy()

        for b in range(max_indices.shape[0]):
            raw = []
            prev = -1
            for t in range(lengths[b]):
                idx = max_indices[b, t]
                if idx != 0 and idx != prev:
                    raw.append(idx_to_char.get(idx, ''))
                prev = idx
            all_texts[start + b] = ''.join(raw)

    return all_texts


# 4. Main pipeline
def pipeline(image_path):
    # 1. Detect cells with grid positions
    detector = TableDetector(YOLO_WEIGHTS)
    cells = detector.get_cells_with_grid_positions(image_path)
    if not cells:
        print("No cells detected.")
        return pd.DataFrame()

    # 2. Deduplicate boxes (spanning cells share the same crop)
    box_to_positions = defaultdict(list)
    for cell in cells:
        box_to_positions[cell['box']].append((cell['row'], cell['col']))
    unique_boxes = list(box_to_positions.keys())

    # 3. Crop images
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    crops = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in unique_boxes]

    # 4. Run OCR (imported from preprocess)
    texts = batch_ocr(crops)
    box_to_text = {box: txt for box, txt in zip(unique_boxes, texts)}

    # 5. Build table using row/col indices
    max_row = max(c['row'] for c in cells)
    max_col = max(c['col'] for c in cells)
    table = [['' for _ in range(max_col+1)] for _ in range(max_row+1)]

    for cell in cells:
        r, c = cell['row'], cell['col']
        txt = box_to_text[cell['box']]
        if not cell['is_spanning'] or table[r][c] == '':
            table[r][c] = txt

    df = pd.DataFrame(table)
    print(df)
    df.to_csv("output_table.csv", index=False)
    return df



# 5. Run
if __name__ == "__main__":
    pipeline("E:/PROJECTS/OCR/Data/table_image.jpg")   # replace with your test image