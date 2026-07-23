import argparse
import pandas as pd
from inferencePipeline.tableDetector import TableDetector
from inferencePipeline.wordDetector import load_crnn_model, batch_ocr, predict_single_word
from inferencePipeline.cropImages import crop_and_save_cells

def pipeline(image_path, yolo_weights, crnn_weights, output_csv):
    print("1. Loading Models...")
    detector = TableDetector(yolo_weights)
    crnn_model = load_crnn_model(crnn_weights)

# detecting the cells are present or not in the input image 

    print(f"2. Detecting Table Layout in '{image_path}'...")
    cells, img = detector.get_cells(image_path)
    
    if not cells:
        print("No cells detected! Check YOLO model or image.")
        img_height, img_width = img.shape[:2]
        
        # Create a single cell that covers the whole image at Row 0, Col 0
        cells = [{
            'row': 0,
            'col': 0,
            'box': (0, 0, img_width, img_height)
        }]

# Cropping the cells that are found

    print(f"3. Cropping {len(cells)} cells and running CRNN OCR...")
    saved_crops_data = crop_and_save_cells(img, cells, output_dir="extracted_table_cells")

    # Use the imported function to crop and save to a folder named extracted_table_cells
    crops = [item["image_matrix"] for item in saved_crops_data]

    print("Running the CRNN OCR on crops.....")
    texts = []
    for crop in crops:
        predicted_text = predict_single_word(crnn_model, crop)
        texts.append(predicted_text)

# Below code is used to reconstruct the words in the terminal and in the CSV files
    print("4. Reconstructing Table and saving to CSV...")
    
    # Filter out spanning cells (which have None for row/col) to find grid dimensions
    grid_cells = [c for c in cells if c.get('row') is not None and c.get('col') is not None]

    if grid_cells:
        # Find max rows and columns to create our empty 2D array
        max_row = max(c['row'] for c in grid_cells)
        max_col = max(c['col'] for c in grid_cells)
    else:
        max_row, max_col = 0, 0

    table_data = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    # Populate the 2D array with our extracted text
    spanning_texts = []
    for cell, text in zip(cells, texts): #type: ignore
        r, c = cell.get('row'), cell.get('col')
        # If it's a standard grid cell
        if r is not None and c is not None:
            table_data[r][c] = text.strip()
        # If it's a spanning cell
        else:
            spanning_texts.append(text.strip())

    # Convert to Pandas DataFrame and Save
    df = pd.DataFrame(table_data)

    print("\n" + "="*50)
    print(df.to_string()) # Print the dataframe to terminal
    print("="*50)

    if spanning_texts:
        print(f"\nNote: Found spanning cells with text: {spanning_texts}")

    df.to_csv(output_csv, index=False, header=False) # No headers, just raw data
    print(f"\n SUCCESS: Table saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Table OCR Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--yolo", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--crnn", required=True, help="Path to CRNN weights (.pth)")
    parser.add_argument("--output", default="output_table.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    pipeline(args.image, args.yolo, args.crnn, args.output)