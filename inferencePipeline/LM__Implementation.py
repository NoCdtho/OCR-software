import pytesseract
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def extract_text_from_image(image_path):
    """Extracts raw text from an image using Tesseract."""
    image = Image.open(image_path)
    raw_text = pytesseract.image_to_string(image)
    return raw_text.strip()

# Using a BART model fine-tuned for text/grammar correction
MODEL_NAME = "pszemraj/bart-base-grammar-synthesis"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def chunk_text(text):
    """Splits text into rough sentence chunks to avoid token limits."""
    # A simple split by period. For production, consider NLTK's sent_tokenize.
    sentences = text.replace('\n', ' ').split('. ')
    return [s.strip() + '.' for s in sentences if len(s.strip()) > 0]

def correct_ocr_text(raw_text, max_length=128):
    """Passes raw OCR text through BART for error correction."""
    chunks = chunk_text(raw_text)
    cleaned_chunks = []
    
    for chunk in chunks:
        # 1. Tokenize the input text
        inputs = tokenizer(
            chunk, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True
        ).to(device)
        
        # 2. Generate the corrected text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4, # Beam search improves quality
                early_stopping=True
            )
        
        # 3. Decode the output back to a string
        cleaned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_chunks.append(cleaned_text)
        
    # Stitch the corrected document back together
    return " ".join(cleaned_chunks)