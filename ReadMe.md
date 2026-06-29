# OCR-Software

Building a OCR software for image text and tablular text along with table recognition.


Training CRNN models for text recognition.
Training YOLO models for table and table cell recognition.
 
How the CRNN is built to scan the hadwritten document into machine readable format:
 >used IAM dataset training on CRNN model for handwritten text recognition.
 >Used mjsynth dataset for yolo model to detect

Training YOLO models to scan the table and table cells into machine readable format:
 >used Pubtables-1m to detect printed tables and cells.
>used publaynet to detect tables in a image(not necessary).
>Training further to detect handwritten tables in a image.
