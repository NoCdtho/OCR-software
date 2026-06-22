import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2 
from typing import List, Tuple
import numpy as np
from datasets import load_dataset

ds = load_dataset(
    "priyank-m/MJSynth_text_recognition",
     split= "train",
     streaming=True
)
ds = ds.take(5000)
unique_chars = sorted(list(set("".join(ds["label"])))) # Takes whole row of label feature and extracts the unique characters and stores in form of list.
my_dictionary = {char: idx for idx, char in enumerate(unique_chars, start=1)} # Build the dictionary with index 0 reserved for the CTC token.

# Reverse mapping for decoding (0 = blank)
idx_to_char = {0: '-'}  # blank, not printed
idx_to_char.update({v: k for k, v in my_dictionary.items()})

# Your exact transform pipeline from training
my_transform_pipeline = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((32, 128)),   # (height, width)
    T.ToTensor()
])


# 2. Crop bounding boxes from the original image

def crop_cells(image_path_or_array, bboxes: List[Tuple[int,int,int,int]]):
    """
    Crops the image at each bounding box.
    bboxes: list of (x1, y1, x2, y2) in absolute pixel coordinates.
    Returns a list of PIL Images.
    """
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('RGB')
    elif isinstance(image_path_or_array, np.ndarray):
        # OpenCV BGR -> RGB PIL
        img = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("Input must be filepath or numpy array")

    crops = []
    w_img, h_img = img.size
    for (x1, y1, x2, y2) in bboxes:
        # Clip to image boundaries
        x1 = max(0, min(int(x1), w_img-1))
        y1 = max(0, min(int(y1), h_img-1))
        x2 = max(x1+1, min(int(x2), w_img))
        y2 = max(y1+1, min(int(y2), h_img))
        crop = img.crop((x1, y1, x2, y2))
        crops.append(crop)
    return crops


# 3. Dataset for inference (applies your training transform)
class CellRecognitionDataset(Dataset):
    def __init__(self, cropped_images: List[Image.Image], transform):
        self.images = cropped_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # The transform expects a PIL image (already RGB, grayscale conversion inside)
        tensor = self.transform(img)   # shape (1, 32, 128)
        return tensor, idx   # return index to keep order after batching


# 4. Collate function (simple stacking, all images are the same size)
def simple_collate(batch):
    tensors, indices = zip(*batch)
    images = torch.stack(tensors, dim=0)   # (B, 1, 32, 128)
    return images, list(indices)


# 5. Greedy CTC decoding using your dictionary
def greedy_decode(logits: torch.Tensor, idx_to_char: dict) -> List[str]:
    """
    logits: (B, T, nclass) output from CRNN (before softmax)
    Returns list of decoded strings.
    """
    log_probs = F.log_softmax(logits, dim=2)   # (B, T, nclass)
    _, max_indices = torch.max(log_probs, dim=2)  # (B, T)
    decoded_texts = []
    for seq in max_indices:
        # Collapse repeated characters and remove blanks
        text = []
        prev = -1
        for token in seq.tolist():
            if token != 0 and token != prev:   # 0 is CTC blank
                text.append(idx_to_char.get(token, ''))
            prev = token
        decoded_texts.append(''.join(text))
    return decoded_texts


# 6. Batch inference function
def batch_recognize_cells(model, image_path, bboxes, device, batch_size=8):
    """
    model: your CRNN model (in eval mode, on device)
    image_path: path to table image (or numpy array)
    bboxes: list of (x1,y1,x2,y2)
    device: torch device
    Returns a list of extracted strings (same order as bboxes).
    """
    # Crop
    crops = crop_cells(image_path, bboxes)

    # Create dataset & loader
    dataset = CellRecognitionDataset(crops, my_transform_pipeline)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=simple_collate)

    all_texts = [""] * len(dataset)  # placeholder for reordering

    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            logits = model(images)  # expected shape (B, T, nclass)

            # If your model outputs (T, B, nclass), permute it
            if logits.dim() == 3 and logits.size(1) == images.size(0):
                logits = logits.permute(1, 0, 2)

            decoded = greedy_decode(logits, idx_to_char)
            for idx, text in zip(indices, decoded):
                all_texts[idx] = text

    return all_texts


