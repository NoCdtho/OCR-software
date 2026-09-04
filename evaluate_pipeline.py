import jiwer
from main import pipeline # Import your existing pipeline logic

def calculate_metrics(ground_truth_list, predicted_list):
    """
    Calculates CER and WER. 
    Expects lists of strings (e.g., all words in a document).
    """
    # Join lists into single document-level strings for standard evaluation
    truth_doc = " ".join(ground_truth_list)
    pred_doc = " ".join(predicted_list)

    wer = jiwer.wer(truth_doc, pred_doc)
    cer = jiwer.cer(truth_doc, pred_doc)
    
    return cer, wer

# 1. These are the exact, correct words from your test image
ground_truth_words = ["corresponding", "composition", "we", "move", "to", "stop"]

# 2. These are the words your pipeline outputs (the 'texts' list in main.py)
# (You will need to modify main.py slightly to return 'texts' instead of just saving the CSV)
predicted_words_no_bart = ["correspouding", "composition", "us", "MOVE", "Trop", "stop"]
predicted_words_with_bart = ["corresponding", "composition", "we", "MOVE", "Drop", "stop"] 

# Calculate Baseline (No BART)
base_cer, base_wer = calculate_metrics(ground_truth_words, predicted_words_no_bart)
print(f"Baseline CRNN -> CER: {base_cer*100:.2f}%, WER: {base_wer*100:.2f}%")

# Calculate Final (With BART)
final_cer, final_wer = calculate_metrics(ground_truth_words, predicted_words_with_bart)
print(f"Final Pipeline -> CER: {final_cer*100:.2f}%, WER: {final_wer*100:.2f}%")