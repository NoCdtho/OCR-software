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

        # Create a copy of the image to draw bounding boxes on
        annotated_img = img.copy()

        # Run inference
        results = self.model(img, conf=conf, iou=iou)
        rows, cols = [], []

        # Parse YOLO predictions
        for result in results:
            if result.boxes is None: #type: ignore
                continue
            for box in result.boxes: #type: ignore
                cls_id = int(box.cls) # box.cls is a class_ID tensor used to return the category prediction(what is this)
                class_name = self.class_names.get(cls_id, "")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) # xyxy returns the coordinates of the bounding box
                
                if class_name == "table row":
                    rows.append((x1, y1, x2, y2)) 
                elif class_name == "table column":
                    cols.append((x1, y1, x2, y2))

        # Sort spatially
        rows.sort(key=lambda r: r[1])  # Top to bottom
        cols.sort(key=lambda c: c[0])  # Left to right

        cells = []

        # 2. Calculate intersections for standard grid cells and draw them (Green)
        for r_idx, (rx1, ry1, rx2, ry2) in enumerate(rows):
            for c_idx, (cx1, cy1, cx2, cy2) in enumerate(cols):
                # Find the overlapping rectangle between row and column
                ix1 = max(rx1, cx1)
                iy1 = max(ry1, cy1)
                ix2 = min(rx2, cx2)
                iy2 = min(ry2, cy2)

                print(f"coordinates for the identified cell: {ix1}, {iy1}, {ix2} and {iy2}")

                # If a valid intersection exists (width and height > 0)
                if ix1 < ix2 and iy1 < iy2:
                    cells.append({
                        "row": r_idx,
                        "col": c_idx,
                        "box": (ix1, iy1, ix2, iy2),
                        "type": "standard"
                    })
                    
                    # Draw green bounding box for standard cells: (B, G, R) -> (0, 255, 0)
                    cv2.rectangle(annotated_img, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

        # Return the extracted cell data and the newly annotated image
        return cells, annotated_img
    
if __name__ == "__main__":
    WEIGHTS_PATH = "E:/PROJECTS/APT_Summer_Project/TrainedModelsWeights/yoloPubtables_1M.pt"
    IMAGE_PATH = "E:/PROJECTS/APT_Summer_Project/Pipeline/ImageCreation/synthetic_iam_table_cells_only_fixed.jpg"
    OUTPUT_PATH = "annoted_result.jpg"

    print("Loading models.....")
    detector = TableDetector(weights_path=WEIGHTS_PATH)

    print("Scanning image.....")
    cells, annoted_img = detector.get_cells(image_path=IMAGE_PATH)

    cv2.imwrite(OUTPUT_PATH, annoted_img)
    print(f"Success found the number of cells are: {len(cells)}")
    print(f"Saved the annotated image to: {OUTPUT_PATH}")