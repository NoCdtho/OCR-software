import cv2
import os
from inferencePipeline.tableDetector import TableDetector

"""
    Takes the original image and the list of detected cells,
    crops them, and saves them to an output directory.
"""

def crop_to_exact_word(cell_image, padding=8):
    """
    Analyzes a cropped cell, ignores the outer border lines, 
    and returns a sub-crop tightly wrapped around the actual text.
    """
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    # Binarize: ink becomes white, background becomes black
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = cell_image.shape[:2]
    valid_contours = []
    
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        
        # 1. Identify table lines: Shapes touching the edge AND spanning most of the cell
        touches_edge = (x <= 5) or (y <= 5) or (x + cw >= w - 5) or (y + ch >= h - 5)
        is_long_line = (cw > w * 0.75) or (ch > h * 0.75)
        
        if touches_edge and is_long_line:
            continue # Skip this border line
        
        # 2. Filter out tiny background noise/specks
        if cw * ch < 15:
            continue
            
        valid_contours.append(c)
        
    # Fallback: If no valid text is found, return the original cell to prevent crashes
    if not valid_contours:
        return cell_image 
        
    # 3. Find the global bounding box for the remaining text contours
    min_x = min(cv2.boundingRect(c)[0] for c in valid_contours)
    min_y = min(cv2.boundingRect(c)[1] for c in valid_contours)
    
    max_x = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours)
    max_y = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours)
    
    # 4. Apply padding and boundary checks
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w, max_x + padding)
    max_y = min(h, max_y + padding)
    
    # 5. Return the tightly cropped word
    return cell_image[min_y:max_y, min_x:max_x]

def crop_and_save_cells(img, cells, output_dir="cropped_cells"):

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

        crop = crop_to_exact_word(crop)

        # 3. Generate a meaningful filename
        if cell.get("type") == "standard":
            # e.g., "cell_r0_c2.jpg" for Row 0, Column 2
            filename = f"cell_row{cell['row']}_column{cell['col']}.jpg"
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

def remove_borders(image):
    """
    Isolates text by finding stray borders touching the edges of the crop 
    and painting them white, leaving central text untouched.
    """
    # 1. Convert to grayscale and binarize (text/lines become white, background black)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Find contours of all the drawn elements
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image.shape[:2]
    cleaned_image = image.copy()
    
    # Define a 10% margin to check if an object touches the absolute edges
    margin_x = max(5, int(w * 0.10))
    margin_y = max(5, int(h * 0.10))
    
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        
        # Check if the object touches the outer margins of the crop
        touches_left = x <= margin_x
        touches_right = (x + cw) >= (w - margin_x)
        touches_top = y <= margin_y
        touches_bottom = (y + ch) >= (h - margin_y)
        
        # Condition 1: Vertical border (Touches left/right AND is relatively tall)
        is_vertical_border: bool = (touches_left or touches_right) and (ch > h * 0.5) 
        
        # Condition 2: Horizontal border (Touches top/bottom AND is relatively wide)
        is_horizontal_border = (touches_top or touches_bottom) and (cw > w * 0.5)
        
        if is_vertical_border or is_horizontal_border:
            # Paint over the border with white
            cv2.drawContours(cleaned_image, [c], -1, (255, 255, 255), thickness=cv2.FILLED)
            # Add a slight extra thickness to catch grey anti-aliasing pixels on the edges
            cv2.drawContours(cleaned_image, [c], -1, (255, 255, 255), thickness=4)
            
    return cleaned_image

def main():
    # File paths
    WEIGHTS_PATH = "E:/PROJECTS/OCRSoftware/Server/TrainedModelsWeights/yoloPubtables_1M.pt"
    IMAGE_PATH = "E:/PROJECTS/OCRSoftware/TestImage/TableImages/t10.jpg"
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

