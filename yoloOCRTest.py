# main.py
import cv2
import argparse
from tableDetector import TableDetector

def main():
    # Setup terminal arguments
    parser = argparse.ArgumentParser(description="Detect table cells using a trained YOLO model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--weights", type=str, required=True, help="Path to your YOLO .pt model weights.")
    args = parser.parse_args()

    print(f"Loading YOLO model from '{args.weights}'...")
    detector = TableDetector(args.weights)
    
    print(f"Processing image '{args.image}'...")
    cells, img = detector.get_cells(args.image)
    
    # --------------------------------------------------
    # PRINT THE NUMBER OF DETECTED CELLS TO TERMINAL
    # --------------------------------------------------
    print("\n" + "="*50)
    print(f" SUCCESS: Detected {len(cells)} table cells!")
    print("="*50 + "\n")

    # Draw green bounding boxes around all detected cells
    for (x1, y1, x2, y2) in cells:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save the visualization to confirm it worked correctly
    output_filename = "detected_cells_output.jpg"
    cv2.imwrite(output_filename, img)
    print(f"Saved visualization to: '{output_filename}'")

if __name__ == "__main__":
    main()