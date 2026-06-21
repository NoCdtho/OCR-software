import cv2
from ultralytics import YOLO

# Your model's class mapping
CLASS_MAP = {
    "table": 0,
    "table row": 1,
    "table column": 2,
    "table spanning cell": 3,
}

def detect_table_cells(image_path, model_path, conf=0.25, iou=0.45):
    """
    Detect all 'table spanning cell' objects in the full-page image.
    Returns list of dicts with absolute pixel coordinates.
    """
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(img, conf=conf, iou=iou)
    # ultralytics returns Result object and .boxes is the attribute in the object.
    cells = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]   # e.g., "table spanning cell"

            # Only keep cells (adjust if your model also has a "table cell" class)
            if class_name not in ["table spanning cell", "table cell"]:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cells.append({
                "box_xyxy": [int(round(x1)), int(round(y1)),
                             int(round(x2)), int(round(y2))],
                "confidence": round(float(box.conf[0]), 4),
            })

    print(f"Total table cells detected: {len(cells)}")
    return cells


def sort_cells_by_position(cells, method="top-left"):
    """
    Simple row-wise sorting (top-left to bottom-right).
    For more accurate table reconstruction, you could use the row/column detections
    to group cells by row.
    """
    if method == "top-left":
        # Sort by y1, then x1
        return sorted(cells, key=lambda c: (c["box_xyxy"][1], c["box_xyxy"][0]))
    return cells


if __name__ == "__main__":
    MODEL_PATH = "E:/PROJECTS/OCR/Server/best.pt"
    IMAGE_PATH = "E:/PROJECTS/OCR/Server/detected_cells.jpg"

    cells = detect_table_cells(IMAGE_PATH, MODEL_PATH, conf=0.3)

    # Sort to preserve reading order
    cells_sorted = sort_cells_by_position(cells)

    # Print first few
    for i, cell in enumerate(cells_sorted[:10]):
        print(f"Cell {i+1}: {cell['box_xyxy']}  conf={cell['confidence']:.3f}")