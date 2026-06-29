import argparse
import pandas as pd
from inferencePipeline.tableDetector import TableDetector
from inferencePipeline.wordDetector import load_crnn_model, batch_ocr

def pipeline(image_path, yolo_weights, crnn_weights, output_csv):
    print("1. Loading Models...")
    detector = TableDetector(yolo_weights)
    crnn_model = load_crnn_model(crnn_weights)

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

    print(f"3. Cropping {len(cells)} cells and running CRNN OCR...")
    crops = []
    for cell in cells:
        x1, y1, x2, y2 = cell['box']
        # OpenCV cropping is img[y_start:y_end, x_start:x_end]
        crop = img[y1:y2, x1:x2]
        crops.append(crop)

    # Run batched OCR
    texts = batch_ocr(crnn_model, crops, batch_size=32)

    print("4. Reconstructing Table and saving to CSV...")
    # Find max rows and columns to create our empty 2D array
    max_row = max(c['row'] for c in cells)
    max_col = max(c['col'] for c in cells)
    table_data = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    # Populate the 2D array with our extracted text
    for cell, text in zip(cells, texts):
        r, c = cell['row'], cell['col']
        table_data[r][c] = text.strip()

    # Convert to Pandas DataFrame and Save
    df = pd.DataFrame(table_data)
    df.to_csv(output_csv, index=False, header=False) # No headers, just raw data
    
    print("\n" + "="*50)
    print(df.to_string()) # Print the dataframe to terminal
    print("="*50)
    print(f"\n SUCCESS: Table saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Table OCR Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--yolo", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--crnn", required=True, help="Path to CRNN weights (.pth)")
    parser.add_argument("--output", default="output_table.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    pipeline(args.image, args.yolo, args.crnn, args.output)