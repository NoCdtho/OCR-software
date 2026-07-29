# Deep Learning-Based Table Detection and Handwritten OCR

An end-to-end multi-stage Optical Character Recognition (OCR) pipeline designed to extract, recognize, and structure tabular data and handwritten text from document images. 

The pipeline leverages a combination of computer vision and natural language processing (NLP) to detect table cells, extract text via a custom CRNN, and correct prediction errors using a BART transformer before reconstructing the data into a structured CSV format.

##  Pipeline Architecture

This project processes scanned documents through five distinct stages:

1. **Detection and Localization (YOLO):** Identifies tables and bounds individual cells within the document image.
2. **Extraction (OpenCV):** Isolates and crops the detected cells based on the YOLO bounding box coordinates.
3. **Character Recognition (Custom CRNN):** Processes the cropped cell images through a Convolutional Recurrent Neural Network feature extractor to digitize the text.
4. **Post-Processing & Error Correction (BART):** Passes the raw CRNN output through a BART transformer to correct spelling and contextual errors based on natural language understanding.
5. **Data Reconstruction (Pandas/NumPy):** Maps the corrected text back to its original spatial layout (rows and columns) to generate a final, structured CSV file.

##  Tech Stack & Models

### Core Frameworks
* **PyTorch:** Foundational deep learning engine for model training and inference.
* **Ultralytics:** YOLO model implementation and training logic.
* **Transformers (Hugging Face):** BART model integration.

### Models & Datasets
* **Detection:** YOLO (trained on the **PubLayNet** dataset).
* **Recognition:** Custom CRNN (trained on the **MJSynth** dataset).
* **Error Correction:** BART transformer.
* **Environment:** Models were trained using Kaggle notebooks.

### Data Processing
* **Computer Vision:** `opencv-python` (cv2), `Pillow` (PIL)
* **Data Structuring:** `pandas`, `numpy`

### Working procedure
1. we can create table of image using the ImageCreation/IAMtable.py file.
2. Then we can execute the main.py file using the image generated and both the models. 