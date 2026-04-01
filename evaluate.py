import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import PianoTranscriptionCRNN
from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculates TP, FP, FN, TN for a batch of predictions.
    """
    # Convert logits to binary predictions (0 or 1)
    preds = (torch.sigmoid(predictions) > threshold).int()
    targets = targets.int()
    
    # Flatten tensors to compare every single frame independently
    p = preds.flatten()
    t = targets.flatten()
    
    tp = torch.sum((p == 1) & (t == 1)).item()
    fp = torch.sum((p == 1) & (t == 0)).item()
    fn = torch.sum((p == 0) & (t == 1)).item()
    tn = torch.sum((p == 0) & (t == 0)).item()
    
    return tp, fp, fn, tn

def print_model_stats(model):
    """Calculates and prints the model's parameter count and physical size."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate physical size in Megabytes (Parameters + Buffers)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    
    print("\n" + "-"*40)
    print("MODEL ARCHITECTURE STATS")
    print("-"*40)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size:           {size_all_mb:.2f} MB")
    print("-"*40 + "\n")

def evaluate_model(model_path, maestro_dir, is_quantized=False):
    # 1. Force CPU if evaluating a quantized model
    if is_quantized:
        print("\n[*] Quantized model detected. Forcing evaluation on CPU.")
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print("Loading test dataset...")
    preprocessor = MaestroPreprocessor().to(device)
    metadata = load_metadata(maestro_dir)
    
    test_dataset = MaestroDataset(
        maestro_dir=maestro_dir, 
        target_sr=preprocessor.target_sr, 
        hop_length=preprocessor.hop_length, 
        metadata=metadata["test"]
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # 2. Initialize the standard architecture
    print(f"Initializing base model architecture on {device.upper()}...")
    model = PianoTranscriptionCRNN(freq_bins=88).to(device)
    
    # 3. Transform the architecture to match the compressed weights
    if is_quantized:
        print(f"Loading compiled TorchScript model from {model_path}...")
        # TorchScript models already contain their architecture! 
        # No need to initialize PianoTranscriptionCRNN or apply quantization here.
        model = torch.jit.load(model_path, map_location=device)
        
    else:
        print(f"Initializing standard model architecture on {device.upper()}...")
        model = PianoTranscriptionCRNN(freq_bins=88).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
    model.eval()
    
    print_model_stats(model)

    # Track totals
    total_onset_tp, total_onset_fp, total_onset_fn, total_onset_tn = 0, 0, 0, 0
    total_frame_tp, total_frame_fp, total_frame_fn, total_frame_tn = 0, 0, 0, 0
    
    print("Starting evaluation...")
    with torch.no_grad(): # No gradients needed for grading!
        for waveforms, frame_labels, onset_labels in tqdm(test_dataloader, desc="Grading Batches"):
            waveforms = waveforms.to(device)
            frame_labels = frame_labels.to(device)
            onset_labels = onset_labels.to(device)
            
            # Preprocess
            if waveforms.dim() == 3: waveforms = waveforms.squeeze(1) 
            spectrograms = preprocessor(waveforms, orig_sr=test_dataset.sample_rate, augment=False)
            if spectrograms.dim() == 3: spectrograms = spectrograms.unsqueeze(1)
            
            # Align time dimensions
            min_time = min(spectrograms.shape[-1], frame_labels.shape[-1], onset_labels.shape[-1])
            spectrograms = spectrograms[:, :, :, :min_time]
            frame_labels = frame_labels[:, :, :min_time]
            onset_labels = onset_labels[:, :, :min_time]
            
            # Forward pass
            onset_logits, frame_logits = model(spectrograms)
            
            # Calculate metrics for Onsets
            tp, fp, fn, tn = calculate_metrics(onset_logits, onset_labels)
            total_onset_tp += tp; total_onset_fp += fp; total_onset_fn += fn; total_onset_tn += tn
            
            # Calculate metrics for Frames
            tp, fp, fn, tn = calculate_metrics(frame_logits, frame_labels)
            total_frame_tp += tp; total_frame_fp += fp; total_frame_fn += fn; total_frame_tn += tn

    # --- Final Math ---
    def compute_scores(tp, fp, fn, tn):
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return precision, recall, f1, accuracy

    o_p, o_r, o_f1, o_a = compute_scores(total_onset_tp, total_onset_fp, total_onset_fn, total_onset_tn)
    f_p, f_r, f_f1, f_a = compute_scores(total_frame_tp, total_frame_fp, total_frame_fn, total_frame_tn)

    print("\n" + "="*40)
    print("FINAL MODEL EVALUATION SCORES")
    print("="*40)
    print(f"--- FRAME METRICS (Sustained Notes) ---")
    print(f"F1-Score:  {f_f1 * 100:.2f}%")
    print(f"Precision: {f_p * 100:.2f}%")
    print(f"Recall:    {f_r * 100:.2f}%")
    print(f"Accuracy:  {f_a * 100:.2f}%  <-- (Misleading due to empty space)")
    print("")
    print(f"--- ONSET METRICS (Hammer Strikes) ---")
    print(f"F1-Score:  {o_f1 * 100:.2f}%")
    print(f"Precision: {o_p * 100:.2f}%")
    print(f"Recall:    {o_r * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    MAESTRO_DIR = r"T:/Uni/Final year/Digital Winter Sem/Final Project/maestro-v3.0.0/maestro-v3.0.0_22050Hz"
    
    # --- Test the Standard Model ---
    # STANDARD_MODEL_PATH = r"trained_models/onsets_1.pth"
    # evaluate_model(STANDARD_MODEL_PATH, MAESTRO_DIR, is_quantized=False)
    
    # --- Test the Compressed Model ---
    COMPRESSED_MODEL_PATH = r"trained_models/onsets_2_pq.pt"
    evaluate_model(COMPRESSED_MODEL_PATH, MAESTRO_DIR, is_quantized=True)