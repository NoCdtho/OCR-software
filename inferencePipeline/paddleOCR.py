import os
# Force disable oneDNN at the engine level before importing paddle modules
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'

from paddleocr import PaddleOCR

# Explicitly set enable_mkldnn=False to bypass the PIR layout bug

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

def image_to_word(image_path):
    result = ocr.predict(image_path)

    words = []

    for res in result:
        data = res.json

        if "res" in data:
            texts = data["res"].get("rec_texts", [])
            words.extend(texts)

    return " ".join(words).strip()
