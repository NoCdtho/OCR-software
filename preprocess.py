"""

CRNN model loading + batch_ocr

This module handles text recognition using a trained CRNN model:
    Loads the model once, moves it to the appropriate device, and sets it to evaluation mode.
Preprocessing: Converts each cropped table-cell image to grayscale, resizes it to a fixed height 
    (keeping aspect ratio with a maximum width), and turns it into a normalized tensor.
Batch inference: Groups multiple crops together, pads them to the same width, and runs them through the CRNN. 
                The model outputs CTC log-probabilities, which are decoded using greedy CTC 
                (removing blanks and repeated characters) to produce the final text strings.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from CRNN.CRNN_model import CRNN  # import the class, not an alias


#  Configuration – adapt to your training setup

TARGET_HEIGHT = 32
MAX_WIDTH = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Replace with your actual IAM character set (order must match training)
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"-!?"
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank


#  Load your trained CRNN model once (do this at module level or in main)

# Adjust the constructor arguments to match your CRNN class definition!
crnn_model = CRNN(num_classes=NUM_CLASSES)  # add other params if needed
crnn_model.load_state_dict(torch.load("E:/PROJECTS/OCR/Trained models/IAMCRNN.pth", map_location=DEVICE))
crnn_model.to(DEVICE)
crnn_model.eval()


#  Preprocessing
def preprocess_crop(crop_img: np.ndarray) -> torch.Tensor:
    """
    Convert a BGR crop (H,W,3) to a grayscale tensor (1, 32, W_padded).
    """
    # Convert BGR to grayscale PIL
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
    w, h = pil_img.size
    new_h = TARGET_HEIGHT
    new_w = int(w * (TARGET_HEIGHT / h))
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH
    # Resize (use Resampling.BILINEAR – not deprecated Image.BILINEAR)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    # Convert to tensor (1, new_h, new_w) and normalise to [0,1]
    img_tensor = F.to_tensor(pil_img)
    # If your training used mean/std normalisation, apply it here:
    # img_tensor = (img_tensor - MEAN) / STD
    return img_tensor

#  Batch OCR
def batch_ocr(crops: list, batch_size=32):
    """
    crops: list of numpy arrays (BGR crops)
    Returns: list of strings, same order as crops
    """
    all_texts = [""] * len(crops)
    idx_to_char = {i: c for i, c in enumerate(CHARSET, start=0)}  # blank = 0

    for start in range(0, len(crops), batch_size):
        batch_crops = crops[start:start+batch_size]

        # Preprocess all
        tensors = [preprocess_crop(c) for c in batch_crops]
        max_w = max(t.shape[2] for t in tensors)

        # Pad to same width
        padded = torch.zeros(len(tensors), 1, TARGET_HEIGHT, max_w)
        lengths = []
        for j, t in enumerate(tensors):
            _, h, w = t.shape
            padded[j, :, :h, :w] = t
            lengths.append(w)

        padded = padded.to(DEVICE)
        lengths = torch.tensor(lengths, dtype=torch.long)

        with torch.no_grad():
            # Your CRNN should return log_probs (T, B, num_classes)
            log_probs = crnn_model(padded)   # now crnn_model is the instance

        # CTC greedy decoding
        _, max_indices = log_probs.max(dim=2)          # (T, B)
        max_indices = max_indices.permute(1, 0).cpu().numpy()   # (B, T)

        for b in range(max_indices.shape[0]):
            raw = []
            prev = -1
            for t in range(lengths[b]):
                idx = max_indices[b, t]
                if idx != 0 and idx != prev:   # skip blank and repeats
                    raw.append(idx_to_char.get(idx, ''))
                prev = idx
            all_texts[start + b] = ''.join(raw)

    return all_texts