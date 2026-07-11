import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from inferencePipeline.CRNN.CRNN_model import CRNN 

# --- CONFIGURATION ---
TARGET_HEIGHT = 32
MAX_WIDTH = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANT: This CHARSET must match the EXACT characters your new model was trained on.
CHARSET = r""" !"#'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank token (index 0)

def load_crnn_model(weights_path):
    model = CRNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(crop_img: np.ndarray) -> torch.Tensor:
    """Loads an image, converts to grayscale, resizes, and turns into a tensor."""
    if crop_img.size == 0:
        return torch.zeros((1, TARGET_HEIGHT, 1))

    # Convert BGR to grayscale PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
    w, h = pil_img.size

    # Resize keeping aspect ratio based on TARGET_HEIGHT
    new_h = TARGET_HEIGHT
    if h>0:
        new_w = int(w * (TARGET_HEIGHT / h))
    else:
        new_w = TARGET_HEIGHT
    
    new_w = min(new_w, MAX_WIDTH)
    # matching the training interpolation
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    # matcching training range [0, 1]
    img_tensor = F.to_tensor(pil_img)
    return img_tensor

def batch_ocr(model, crops: list, batch_size=32):
    """Runs OCR on a list of numpy image crops in batches."""
    all_texts = [""] * len(crops)
    idx_to_char = {i+1: c for i, c in enumerate(CHARSET)}
    
    for start in range(0, len(crops), batch_size):
        batch_crops = crops[start:start+batch_size]
        tensors = [preprocess_image(c) for c in batch_crops]
        max_w = max(t.shape[2] for t in tensors)
        
        # Pad to max width in this batch
        padded = torch.zeros(len(tensors), 1, TARGET_HEIGHT, max_w)
        for j, t in enumerate(tensors):
            _, h, w = t.shape
            padded[j, :, :h, :w] = t
            
        padded = padded.to(DEVICE)
        
        with torch.no_grad():
            log_probs = model(padded)
            
        # FIXED CTC DECODING 
        _, max_indices = log_probs.max(dim=2)

        # check if CRNN is predicting in batch-first or time-first manner
        if max_indices.shape[0] == len(batch_crops):
            max_indices = max_indices.cpu().numpy()
        else:
            max_indices = max_indices.permute(1, 0).cpu().numpy() 

        T = max_indices.shape[1] # Actual sequence length!
        
        for b in range(max_indices.shape[0]):
            raw = []
            prev = -1
            for t in range(T):
                idx = max_indices[b, t]
                if idx != 0 and idx != prev:
                    raw.append(idx_to_char.get(idx, ''))
                prev = idx
            all_texts[start + b] = ''.join(raw)
            
    return all_texts
