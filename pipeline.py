import torch
from PIL import Image
from torch.utils.data import DataLoader

# Import YOLO components
from YOLO.tableDetector import TableDetector  

# Import CRNN components
from CRNN.CRNN_model import CRNN
from CRNN.preprocess import (
    CellRecognitionDataset, 
    my_transform_pipeline, 
    simple_collate, 
    greedy_decode, 
    idx_to_char
)

def run_pipeline(image_path, yolo_weights, crnn_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    
    # STEP 1: YOLO Detection (Get Coordinates)
    print("1. Running YOLO to detect cells...")
    yolo_detector = TableDetector(weights_path=yolo_weights)
    bboxes = yolo_detector.get_bboxes(image_path, ) # Expected: List of (x1, y1, x2, y2)
    
    if not bboxes:
        print("No cells detected by YOLO.")
        return []

  
    # STEP 2: Pipeline Cropping
    print(f"2. Pipeline cropping {len(bboxes)} images based on coordinates...")
    # Open the original image once
    original_img = Image.open(image_path).convert('RGB')
    cropped_images = []
    
    for (x1, y1, x2, y2) in bboxes:
        # Crop the image at the YOLO coordinates
        crop = original_img.crop((x1, y1, x2, y2))
        cropped_images.append(crop)

    # STEP 3: CRNN Preprocessing & Detection
    print("3. Loading CRNN and running text recognition...")
    # Initialize Model
    crnn_model = CRNN(input_channels=1, hidden_size=256, num_layers=2, num_classes=37)
    crnn_model.load_state_dict(torch.load(crnn_weights, map_location=device))
    crnn_model.to(device)
    crnn_model.eval()

    # Pass the raw cropped images to your existing Dataset 
    # (my_transform_pipeline handles the grayscale, resize, and tensor conversion)
    dataset = CellRecognitionDataset(cropped_images, my_transform_pipeline)
    loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=simple_collate
    )

    extracted_texts = [""] * len(cropped_images)

    # Run Inference
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            logits = crnn_model(images)

            # Re-align dimensions if necessary for CTC decoder
            if logits.dim() == 3 and logits.size(1) == images.size(0):
                logits = logits.permute(1, 0, 2)

            # Decode probabilities into text
            decoded = greedy_decode(logits, idx_to_char)
            
            # Reconstruct the list in the correct order
            for idx, text in zip(indices, decoded):
                extracted_texts[idx] = text

    # FINAL OUTPUT
    final_results = []
    for box, text in zip(bboxes, extracted_texts):
        final_results.append({
            "coordinates": box,
            "text": text
        })
        print(f"Box {box} -> Text: '{text}'")

    return final_results

if __name__ == "__main__":
    IMAGE_PATH = "E:/PROJECTS/OCR/Server/data/detected_cells.jpg"
    YOLO_WEIGHTS = "E:/PROJECTS/OCR/Server/data/best.pt"
    CRNN_WEIGHTS = "E:/PROJECTS/OCR/Server/data/best_crnn_ocr_model.pth"
    
    results = run_pipeline(IMAGE_PATH, YOLO_WEIGHTS, CRNN_WEIGHTS)