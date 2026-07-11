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



if __name__ == "___main___":

    """Takes a word image (either a file path or numpy array), predicts the text 
    using the CRNN model, and draws the prediction on a border above the image.
    """

def detect_and_draw_word(model, image_input):
# 1. Handle both file paths and OpenCV numpy arrays
    if isinstance(image_input, str):
        crop_img = cv2.imread(image_input)
    else:
        crop_img = image_input.copy()

    if crop_img is None or crop_img.size == 0:
        print("Error: Invalid or empty image provided.")
        return None, ""

    # 2. Get the prediction using your existing batch_ocr function
    # We pass it as a list containing a single image
    predicted_texts = batch_ocr(model, [crop_img], batch_size=1)
    text = predicted_texts[0]

    # 3. Create a canvas to display the text without covering the handwriting
    h, w = crop_img.shape[:2]
    
    # Ensure the canvas is wide enough to display the text even if the crop is tiny
    text_size = cv2.getTextSize(f"Pred: {text}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    canvas_w = max(w, text_size[0] + 20)
    canvas_h = h + 45  # Add 45 pixels at the top for the text border

    # Create a white canvas (3 channels for colored text)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # If the input image is grayscale, convert to BGR so we can paste it on the 3-channel canvas
    if len(crop_img.shape) == 2:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)

    # Paste the original cropped image at the bottom of the canvas
    canvas[45:45+h, 0:w] = crop_img

    # 4. Draw the predicted text in red at the top
    cv2.putText(canvas, f"Pred: {text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return canvas, text