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
    # converts the image to grayscale
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    # Binarize: ink becomes white, background becomes black
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # draws a mathematical boundaries i.e. contours around every white shape it finds
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = cell_image.shape[:2]
    valid_contours = []
    
    for c in contours:
        # used to get the exact coordinate of the current shape
        x, y, cw, ch = cv2.boundingRect(c)
        
        # 1. Identify table lines: Shapes touching the edge AND spanning most of the cell
        touches_edge: bool = (x <= 5) or (y <= 5) or (x + cw >= w - 5) or (y + ch >= h - 5)
        is_long_line: bool = (cw > w * 0.75) or (ch > h * 0.75)
        
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

def crop_and_save_cells(img, cells, output_dir="cropped_cells") -> list:

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
        filename = f"cell_row{cell['row']}_column{cell['col']}.jpg"
        save_path = os.path.join(output_dir, filename)
        
        # 4. Save the cropped image to disk
        cv2.imwrite(save_path, crop)
        
        # Store in a list of dictionary in case you want to pass them directly to your CRNN in memory
        cropped_data.append({
            "filepath": save_path,
            "image_matrix": crop,
            "row": cell.get("row"),
            "col": cell.get("col")
        })

    return cropped_data



