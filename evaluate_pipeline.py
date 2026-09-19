import jiwer

def calculate_metrics(ground_truth_list, predicted_list):

    # Join lists into single document-level strings for standard evaluation
    truth_doc = " ".join(ground_truth_list)
    pred_doc = " ".join(predicted_list)

    wer = jiwer.wer(truth_doc, pred_doc)
    cer = jiwer.cer(truth_doc, pred_doc)
    
    return cer, wer