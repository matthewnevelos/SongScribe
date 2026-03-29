import torch
import torch.nn as nn
from format_converter import audio_to_CQT

def evaluate_model(model, test_loader, threshold=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n--- Starting Evaluation on {device} ---")

    model = model.to(device)
    model.eval() 
    
    criterion = nn.BCEWithLogitsLoss()

    # Metric Accumulators
    total_loss = 0.0
    total_tp = 0.0  
    total_fp = 0.0  
    total_fn = 0.0  
    total_tn = 0.0  

    #Evaluation Loop
    with torch.no_grad(): 
        for batch_idx, (waveforms, labels) in enumerate(test_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            if waveforms.dim() == 3:
                if waveforms.shape[0] == 1 and waveforms.shape[1] > 1:
                    waveforms = waveforms.squeeze(0)
                    print("Is this ever used")
                else:
                    waveforms = waveforms.squeeze(1)
                    print("Is this ever used")
                    
            spectrograms = audio_to_CQT(waveforms)
            
            if spectrograms.dim() == 3:
                spectrograms = spectrograms.unsqeeze(1)
                print("Is this ever used")
            
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            total_tp += (preds * labels).sum().item()
            total_fp += (preds * (1 - labels)).sum().item()
            total_fn += ((1 - preds) * labels).sum().item()
            total_tn += ((1 - preds) * (1 - labels)).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Processed batch {batch_idx}...")

    # metric calculations
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