import argparse
import os
from PIL import Image
from CRNN_model import CRNN
import torchvision.transforms as T
import torch

# The index 0 is reserved for the CTC black token '-'
ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz" 

# preprocess of the image
def preprocess_image(image_path):
    image = None
    try:
       image = Image.open(image_path).convert('RGB')
    except IOError:
        print(f"there is a error in {image_path}")
        exit(1)

    transform = T.Compose([
        T.Grayscale(num_output_channels = 1),
        T.Resize((32, 128)),
        T.ToTensor()
    ])
    tensor = transform(image) #type hint

    # Since pytorch handles 4d data it cannot handle a torch of 3 dimensions
    tensor = tensor.unsqueeze(0)  
    return tensor
print(ALPHABET) 


# decode the raw CTC output
def decode_ctc(output_logits, alphabets):
    """I want to extract the index of the higest class probability per timesteps"""
    prediction = output_logits.argmax(dim=1)

    decoded = []
    prev = -1
    for p in prediction:
        p = p.item()
        if p != 0 and p != prev:
            decoded.append(alphabets[p])
        prev = p

    return ''.join(decoded)
 
# Load the trained model
def load_model(checkpoint_path, device='cpu'):
    model = CRNN(input_channels=1,
                 hidden_size=256,
                 num_layers=2,
                 num_classes=len(ALPHABET))
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, device='cpu'):
    # Preprocess
    tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(tensor)  # shape: (1, time_steps, num_classes)
    # Remove batch dimension → (time_steps, num_classes)
    logits = outputs.squeeze(0)

    # Decode
    text = decode_ctc(logits, ALPHABET)
    return text

if __name__ == "__main__":
    # Path to your trained model weights
    CHECKPOINT = "E:/PROJECTS/OCR/Trained models/CRNNmodel.pth"
    # Path to the test image
    IMAGE_PATH = "E:/PROJECTS/OCR/Server/CRNN image test 1.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device='cpu')

    has_nun = any(torch.isnan(p).any() for p in model.parameters())

    result = predict(IMAGE_PATH, model, device='cpu')
    print("Recognised text:", result)