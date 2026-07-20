import torch
from transforms import BartTokenizer, BartModel #type:ignore

# 1. Initialize once at startup
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
LM_Model = BartModel.from_pretrained('facebook/bart-base')

# Move to GPU if available for maximum pipeline throughput
device = "cuda" if torch.cuda.is_available() else "cpu"
LM_Model.to(device)

def process_crnn_batch_with_bart(crnn_text_predictions):
    """
    crnn_text_predictions: List of strings output by your CRNN batch
    Example: ["hello my dog", "is cute", "netowrk alert"]
    """
    if not crnn_text_predictions:
        return None

    # 2. Tokenize the entire batch at once
    inputs = tokenizer(
        crnn_text_predictions, 
        return_tensors="pt", 
        padding=True,          # Pads shorter strings to match the longest one
        truncation=True,       # Ensures strings don't exceed max model length
        max_length=128         # Adjust based on your expected text length
    ).to(device)
    
    # 3. Pass through the base BART model
    with torch.no_grad():       # Disables gradient calculation for faster inference
        outputs = LM_Model(**inputs)
        
    # 4. Extract the hidden states
    # Shape will be: [batch_size, sequence_length, hidden_size]
    last_hidden_states = outputs.last_hidden_state
    
    return last_hidden_states