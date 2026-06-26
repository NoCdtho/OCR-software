import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

# Import your CRNN architecture (adjust the path if needed)
from CRNN.CRNN_model import CRNN 

# --- CONFIGURATION ---
TARGET_HEIGHT = 32
MAX_WIDTH = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANT: This CHARSET must match the EXACT characters your new model was trained on.
CHARSET = r""" !"#'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank token (index 0)

def preprocess_image(image_path: str) -> torch.Tensor:
    """Loads an image, converts to grayscale, resizes, and turns into a tensor."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert BGR to grayscale PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    w, h = pil_img.size

    # Resize keeping aspect ratio based on TARGET_HEIGHT
    new_h = TARGET_HEIGHT
    new_w = int(w * (TARGET_HEIGHT / h))
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH

    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Convert to tensor (1, H, W) and normalize to [0,1]
    img_tensor = F.to_tensor(pil_img)
    
    # Add batch dimension -> (1, 1, H, W) for the model
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def decode_predictions(log_probs: torch.Tensor) -> str:
    """Decodes the model's raw output into a string using CTC greedy decoding."""
    idx_to_char = {i+1: c for i, c in enumerate(CHARSET)}  # blank is 0
    
    # log_probs shape is usually (Sequence_Length, Batch_Size, Num_Classes)
    _, max_indices = log_probs.max(dim=2)                 # (T, B)
    max_indices = max_indices.permute(1, 0).cpu().numpy() # (B, T)
    
    # Get the actual sequence length from the model output
    T = max_indices.shape[1]
    raw_text = []
    prev = -1
    
    # We only have a batch size of 1 here, so we look at index 0
    for t in range(T):
        idx = max_indices[0, t]
        if idx != 0 and idx != prev:   # Skip CTC blanks (0) and repeating characters
            raw_text.append(idx_to_char.get(idx, ''))
        prev = idx
        
    return ''.join(raw_text)

def main():
    parser = argparse.ArgumentParser(description="Read a single word from an image using a trained CRNN.")
    parser.add_argument("--image", type=str, required=True, help="Path to the cropped word image.")
    parser.add_argument("--weights", type=str, required=True, help="Path to your CRNN .pth weights.")
    args = parser.parse_args()

    print(f"Loading CRNN model to {DEVICE}...")
    model = CRNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode

    print(f"Processing image: '{args.image}'")
    tensor = preprocess_image(args.image).to(DEVICE)

    # Run inference without tracking gradients
    with torch.no_grad():
        log_probs = model(tensor)
        
    # Decode text
    recognized_word = decode_predictions(log_probs)
    
    # Print results
    print("\n" + "="*50)
    print(f" DETECTED WORD: {recognized_word}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()