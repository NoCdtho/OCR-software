import os
import argparse
import pandas as pd
from inferencePipeline.tableDetector import TableDetector
from inferencePipeline.wordDetector import load_crnn_model, batch_ocr, predict_single_word
from inferencePipeline.cropImages import crop_and_save_cells
from inferencePipeline.LM_Implementation import correct_ocr_text
from inferencePipeline.deskew import deskew_image
import cv2
import time
import json
import jiwer

def pipeline(image_path, yolo_weights, crnn_weights, output_csv, use_bart=False):
    print("1. Loading Models...")
    detector = TableDetector(yolo_weights)
    print("YOLO model is initialized")
    crnn_model = load_crnn_model(crnn_weights)
    print("CRNN model is initialized")

    # 1.5 Load, Deskew, and Save temporary image
    print("Fixing the image structure...")
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")
        
    deskewed_img = deskew_image(raw_img)
    processed_image_path = "temp_deskewed_image.jpg"
    cv2.imwrite(processed_image_path, deskewed_img)

    # detecting the cells are present or not in the input image 
    print(f"2. Detecting Table Layout in '{processed_image_path}'...")
    cells, img = detector.get_cells(processed_image_path)
    
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

# running the CRNN to detect the words
    print("Running the CRNN OCR on crops.....")
    texts = []
    for crop in crops:
        predicted_text = predict_single_word(crnn_model, crop)

        if use_bart and predicted_text.strip():
            predicted_text = correct_ocr_text(predicted_text)

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
    return texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Table OCR Pipeline")
    parser.add_argument("--image_dir", required=True, help="Path to folder containing test images")
    parser.add_argument("--yolo", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--crnn", required=True, help="Path to CRNN weights (.pth)")
    parser.add_argument("--output", default="output_table.csv", help="Output CSV filename")
    parser.add_argument("--use-bart", action="store_true", help="Enable BART for text correction")
    parser.add_argument("--gt_json", help="Path to ground truth JSON file for CER/WER evaluation", default=None)
    
    args = parser.parse_args()
    
    # Load Ground Truth if provided
    ground_truths = {}
    if args.gt_json:
        try:
            with open(args.gt_json, 'r') as f:
                ground_truths = json.load(f)
            print(f"Loaded ground truth for {len(ground_truths)} images.")
        except Exception as e:
            print(f"Failed to load ground truth JSON: {e}")
            exit()
    
    # Get all images in the directory
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the specified directory.")
        exit()

    processing_times = []
    total_cer_list = []
    total_wer_list = []

    print(f"Starting batch evaluation on {len(image_files)} images...")
    
    for img_name in image_files:
        img_path = os.path.join(args.image_dir, img_name)
        output_csv = f"output_{img_name}.csv"
        
        # Start the timer!
        start_time = time.time()
        
        # Run the pipeline and capture the output texts
        predicted_words_list = pipeline(img_path, args.yolo, args.crnn, output_csv, args.use_bart)
        
        # Stop the timer!
        end_time = time.time()
        
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        processing_times.append(elapsed_time)
        print(f"Processed {img_name} in {elapsed_time:.2f} seconds")
        
        # Calculate CER and WER if ground truth exists for this image
        if ground_truths and img_name in ground_truths:
            true_text = ground_truths[img_name]
            # Join the predicted list of words into a single string for comparison
            pred_text = " ".join([word for word in predicted_words_list if word.strip()])
            
            # Avoid jiwer crash on empty ground truth
            if true_text.strip(): 
                cer = jiwer.cer(true_text, pred_text)
                wer = jiwer.wer(true_text, pred_text)
                total_cer_list.append(cer)
                total_wer_list.append(wer)
                print(f"--> Evaluation for {img_name}: CER={cer*100:.2f}%, WER={wer*100:.2f}%\n")
            else:
                print(f"--> Skipped evaluation for {img_name}: Ground truth is empty.\n")
        else:
            print("\n")

    # Calculate the averages
    total_time = sum(processing_times)
    average_time = total_time / len(processing_times)
    
    print("=" * 50)
    print(f"Total time for {len(image_files)} documents: {total_time:.2f} seconds")
    print(f"Average processing time per document: {average_time:.2f} seconds")
    
    # Print average CER and WER if we evaluated any images
    if total_cer_list and total_wer_list:
        avg_cer = sum(total_cer_list) / len(total_cer_list)
        avg_wer = sum(total_wer_list) / len(total_wer_list)
        print("-" * 50)
        print(f"Average Pipeline CER: {avg_cer*100:.2f}%")
        print(f"Average Pipeline WER: {avg_wer*100:.2f}%")
    
    print("=" * 50)