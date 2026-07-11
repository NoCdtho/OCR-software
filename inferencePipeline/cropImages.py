import cv2
import os
from tableDetector import TableDetector

def crop_and_save_cells(img, cells, output_dir="cropped_cells"):
    """
    Takes the original image and the list of detected cells,
    crops them, and saves them to an output directory.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cropped_data = []

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = cell["box"]
        
        # 1. Safety Check: Ensure coordinates don't exceed image boundaries
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 2. Crop using NumPy slicing [start_y:end_y, start_x:end_x]
        crop = img[y1:y2, x1:x2]
        
        # Skip if the crop is empty (prevents cv2 errors)
        if crop.size == 0:
            continue

        # 3. Generate a meaningful filename
        if cell.get("type") == "standard":
            # e.g., "cell_r0_c2.jpg" for Row 0, Column 2
            filename = f"cell_r{cell['row']}_c{cell['col']}.jpg"
        else:
            # For spanning cells where row/col isn't easily defined
            filename = f"spanning_cell_{i}.jpg"
            
        save_path = os.path.join(output_dir, filename)
        
        # 4. Save the cropped image to disk
        cv2.imwrite(save_path, crop)
        
        # Store in a list in case you want to pass them directly to your CRNN in memory
        cropped_data.append({
            "filepath": save_path,
            "image_matrix": crop,
            "row": cell.get("row"),
            "col": cell.get("col")
        })

    return cropped_data

def main():
    # File paths
    WEIGHTS_PATH = "E:/PROJECTS/OCR/Server/TrainedModelsWeights/yoloPubtables.pt" 
    IMAGE_PATH = "E:/PROJECTS/OCR/TestImage/TableImages/t10.jpg"
    OUTPUT_FOLDER = "extracted_table_cells"

    # Initialize the detector
    print("Loading YOLO model...")
    detector = TableDetector(weights_path=WEIGHTS_PATH)

    # Run inference to get the cells and the annotated image
    print("Scanning image for cells...")
    cells, annotated_img = detector.get_cells(image_path=IMAGE_PATH)

    # We need to reload the pure, unannotated image for cropping 
    # so we don't accidentally include the green/blue YOLO boxes in our crops!
    original_img = cv2.imread(IMAGE_PATH)

    # Crop and save
    print(f"Cropping {len(cells)} cells...")
    saved_crops = crop_and_save_cells(original_img, cells, output_dir=OUTPUT_FOLDER)
    
    print(f"Success! Saved {len(saved_crops)} cropped images into the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    main()