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

def new_Preprocess_image(crop_img: np.ndarray) -> torch.Tensor:
    if crop_img.size == 0:
        return torch.zeros((1, TARGET_HEIGHT, 1))

    # 1. Convert BGR to Grayscale (Keep as NumPy array first for cleaning)
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Shave 5% off top/bottom and 2% off sides to remove stray table border lines
    h_orig, w_orig = gray_img.shape
    margin_y = max(1, int(h_orig * 0.05))
    margin_x = max(1, int(w_orig * 0.02))
    if h_orig > 2 * margin_y and w_orig > 2 * margin_x:
        gray_img = gray_img[margin_y:h_orig-margin_y, margin_x:w_orig-margin_x]

    # 3. Apply Otsu's threshold to force stark black text on a clean white background
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Find where the actual ink is and crop tightly to the word
    inverted = cv2.bitwise_not(binary_img) # Invert so text is white for coordinate finding
    coords = cv2.findNonZero(inverted)
    
    if coords is not None:
        x, y, w_text, h_text = cv2.boundingRect(coords)
        pad = 2 # Small padding so letters don't touch the absolute image edges
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(binary_img.shape[1], x + w_text + pad)
        y_end = min(binary_img.shape[0], y + h_text + pad)
        
        clean_img = binary_img[y_start:y_end, x_start:x_end]
    else:
        clean_img = binary_img # Fallback if cell is empty

    # 5. Convert to PIL Image (Continuing with your original pipeline logic)
    pil_img = Image.fromarray(clean_img)
    w, h = pil_img.size

    # Resize keeping aspect ratio based on TARGET_HEIGHT
    new_h = TARGET_HEIGHT
    if h > 0:
        new_w = int(w * (TARGET_HEIGHT / h))
    else:
        new_w = TARGET_HEIGHT
    
    new_w = min(new_w, MAX_WIDTH)
    
    # Matching your training configuration exactly
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
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


def predict_single_word(model, crop_img: np.ndarray) -> str:
    """Runs OCR on a single numpy image crop without batch padding."""
    # Ensure model is in evaluation mode (CRITICAL for batch size 1)
    model.eval() 
    
    idx_to_char = {i+1: c for i, c in enumerate(CHARSET)}
    
    # 1. Preprocess the image
    img_tensor = new_Preprocess_image(crop_img) # Shape: (1, 32, Width)

    debug_img = F.to_pil_image(img_tensor)
    debug_img.save("model_input_image.png")
    print("Saved debug image check the image")
    
    # 2. Add the Batch Dimension
    # PyTorch models ALWAYS expect 4 dimensions: (Batch, Channel, Height, Width)
    # .unsqueeze(0) changes it from (1, 32, W) to (1, 1, 32, W)
    batch_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # 3. Run Inference
    with torch.no_grad():
        log_probs = model(batch_tensor)
        
    # 4. CTC Decoding for a single sequence
    _, max_indices = log_probs.max(dim=2)

    # Handle batch-first vs time-first output shape from your model
    if max_indices.shape[0] == 1:
        indices_seq = max_indices.cpu().numpy()[0] # Extract the single sequence
    else:
        indices_seq = max_indices.permute(1, 0).cpu().numpy()[0] 

    # 5. Map indices back to characters
    raw = []
    prev = -1
    for idx in indices_seq:
        if idx != 0 and idx != prev:
            raw.append(idx_to_char.get(idx, ''))
        prev = idx
        
    return ''.join(raw)



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