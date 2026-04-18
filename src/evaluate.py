import torch
import torch.nn as nn
from tqdm import tqdm

def decode_onsets_and_frames(onset_probs, frame_probs, onset_threshold=0.5, frame_threshold=0.3):
    """
    Decodes dual-stream probabilities into a binary piano roll.
    Expects 2D tensors of shape [Pitch, Time].
    """
    predictions = torch.zeros_like(frame_probs)
    
    # Iterate through the time dimension
    for t in range(frame_probs.shape[1]):
        # A note is triggered ONLY by the onset head
        new_note = onset_probs[:, t] > onset_threshold
        
        if t == 0:
            predictions[:, t] = new_note.float()
        else:
            # A note sustains if it was ON in the previous frame AND the frame head is still confident
            sustaining_note = (predictions[:, t-1] == 1) & (frame_probs[:, t] > frame_threshold)
            
            predictions[:, t] = (new_note | sustaining_note).float()
            
    return predictions

def binarize_output(frame_probs, onset_probs=None, onset_threshold=0.5, frame_threshold=0.3):
    """
    Convert probabilities to bool based on thresholds.
    """
    
    if onset_probs is not None:
        # Dual-output decoding
        return decode_onsets_and_frames(
            onset_probs=onset_probs, 
            frame_probs=frame_probs, 
            onset_threshold=onset_threshold, 
            frame_threshold=frame_threshold
        )
    else:
        # Simple thresholding for single-output
        return (frame_probs > frame_threshold).float()


def eval_metrics(model, test_loader, preprocessor, activation_function=torch.sigmoid):
    """
    Print Loss, Accuracy, Precision, Recall, F1_Score
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n--- Starting Evaluation on {device} ---")

    model = model.to(device)
    model.eval() 
    
    positive_weight = torch.tensor([15.0]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    # Metric Accumulators
    total_loss = 0.0
    total_tp = torch.tensor(0.0, device=device)
    total_fp = torch.tensor(0.0, device=device)
    total_fn = torch.tensor(0.0, device=device)
    total_tn = torch.tensor(0.0, device=device)

    val_bar = tqdm(test_loader, desc="Evaluating", leave=True)

    #Evaluation Loop
    with torch.no_grad(): 
        for batch in val_bar:
            waveforms = batch[0].to(device)
            targets = [t.to(device) for t in batch[1:]]
            
            if waveforms.dim() == 3 and waveforms.shape[1] > 1:
                waveforms = torch.mean(waveforms, dim=1, keepdim=True)
                
            if waveforms.dim() == 3:
                waveforms = waveforms.squeeze(1)
                    
            spectrograms = preprocessor(waveforms, orig_sr=preprocessor.target_sr, augment=False)
            
            if spectrograms.dim() == 3:
                spectrograms = spectrograms.unsqueeze(1)
                
            min_time = spectrograms.shape[-1]
            for i in range(len(targets)):
                if targets[i].dim() == 2:
                    targets[i] = targets[i].unsqueeze(1)
                min_time = min(min_time, targets[i].shape[-1])
            
            spectrograms = spectrograms[..., :min_time]
            targets = [t[..., :min_time] for t in targets]
            
            outputs = model(spectrograms)
            
            if isinstance(outputs, tuple):
                onset_logits, frame_logits = outputs
                frame_labels, onset_labels = targets[0], targets[1]
                
                onset_loss = criterion(onset_logits, onset_labels)
                frame_loss = criterion(frame_logits, frame_labels)
                loss = onset_loss + frame_loss
                
                eval_labels = frame_labels
                
            else:
                eval_labels = targets[0]
                loss = criterion(outputs, eval_labels)
                
            total_loss += loss.item()

            prediction_logits = frame_logits if isinstance(outputs, tuple) else outputs #type: ignore
            preds = (activation_function(prediction_logits) > 0.5).float()

            preds_flat = preds.reshape(-1)
            labels_flat = eval_labels.reshape(-1)
            

            total_tp += (preds_flat * labels_flat).sum()
            total_fp += (preds_flat * (1 - labels_flat)).sum()
            total_fn += ((1 - preds_flat) * labels_flat).sum()
            total_tn += ((1 - preds_flat) * (1 - labels_flat)).sum()
            
    total_tp = total_tp.item()
    total_fp = total_fp.item()
    total_fn = total_fn.item()
    total_tn = total_tn.item()
    
    eps = 1e-7 
    
    avg_loss = total_loss / max(1, len(test_loader))
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)

    metrics = {
        "Loss": round(avg_loss, 4),
        "Accuracy": round(accuracy * 100, 2), 
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1_Score": round(f1_score * 100, 2)
    }
    
    print("\n--- Evaluation Results ---")
    for key, value in metrics.items():
        suffix = "" if key == "Loss" else "%"
        print(f"{key:>10}: {value}{suffix}")
        
    return metrics  