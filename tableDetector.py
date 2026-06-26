# table_detector.py
import cv2
from ultralytics import YOLO

class TableDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        # Class mapping based on your PubTables-1M training configuration
        self.class_names = {
            0: "table",
            1: "table row",
            2: "table column",
            3: "table spanning cell"
        }

    def get_cells(self, image_path, conf=0.25, iou=0.45):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run inference
        results = self.model(img, conf=conf, iou=iou)
        rows, cols, spanning_cells = [], [], []

        # Parse YOLO predictions
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

        # Sort spatially
        rows.sort(key=lambda r: r[1])  # Top to bottom
        cols.sort(key=lambda c: c[0])  # Left to right

        cells = []

        # 1. Add spanning cells directly to the cell list
        for sp_box in spanning_cells:
            cells.append(sp_box)

        # 2. Calculate intersections for standard grid cells
        for rx1, ry1, rx2, ry2 in rows:
            for cx1, cy1, cx2, cy2 in cols:
                # Find the overlapping rectangle between row and column
                ix1 = max(rx1, cx1)
                iy1 = max(ry1, cy1)
                ix2 = min(rx2, cx2)
                iy2 = min(ry2, cy2)
                
                # If a valid intersection exists (width and height > 0)
                if ix1 < ix2 and iy1 < iy2:
                    
                    # Prevent double-counting: Check if this intersection is inside a spanning cell
                    is_spanning_overlap = False
                    for sx1, sy1, sx2, sy2 in spanning_cells:
                        # Use the center point of the intersection to check if it falls inside a spanning cell
                        center_x, center_y = (ix1 + ix2) / 2, (iy1 + iy2) / 2
                        if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
                            is_spanning_overlap = True
                            break
                    
                    # If it's a normal cell, add it to our list
                    if not is_spanning_overlap:
                        cells.append((ix1, iy1, ix2, iy2))

        return cells, img