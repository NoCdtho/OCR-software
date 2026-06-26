from typing import List, Optional, Tuple
import cv2
from ultralytics import YOLO

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

        rows.sort(key=lambda r: r[1])          # top -> bottom
        cols.sort(key=lambda c: c[0])          # left -> right

        num_rows, num_cols = len(rows), len(cols)

        # Type hint: grid can contain None or a tuple of 4 ints
        grid: List[List[Optional[Tuple[int, int, int, int]]]] = [
            [None for _ in range(num_cols)] for _ in range(num_rows)
        ]
        is_spanning = [[False for _ in range(num_cols)] for _ in range(num_rows)]

        # 1. Place spanning cells
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

        # 2. Fill remaining cells with row/column intersections
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

        # 3. Build flat list in reading order
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