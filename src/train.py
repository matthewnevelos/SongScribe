import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from pathlib import Path
from .evaluate import eval_metrics
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast


def prune_model_step(model, prune_amount=0.1):
    # Applies L1 Unstructured Pruning to Conv2d and Linear layers
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )
    
    
def make_pruning_permanent(model):
    """
    PyTorch pruning uses weight masks during training. 
    This removes the masks and makes the 0.0 weights permanent for saving.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass # Already permanent or wasn't pruned
    print("[*] Pruning masks removed. Zeroed weights are now permanent.")
    

def train_model(model, preprocessor, train_loader, val_loader, epochs=5, lr=1e-4, save_dir="trained_models", 
                model_name=None, augment=True, use_amp=True, prune_per_epoch=0.0, quantize=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Initializing model on {device}...")
    model = model.to(device)
    preprocessor = preprocessor.to(device)
    
    positive_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scaler = torch.amp.grad_scaler.GradScaler("cuda") if use_amp and device=="cuda" else None

    best_val_loss = float('inf')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if model_name is None:
        model_name = model.__class__.__name__
        model_iteration = len(list(save_dir.glob(f"{model_name}*")))
        model_path = save_dir / f"{model_name}_v{model_iteration+1}.pth"
    else:
        model_path = save_dir / model_name
    print(f"Weights will be saved to: {model_path}")

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Training", leave=True)
        
        for batch in train_bar:
            waveforms = batch[0].to(device)
            targets = [t.to(device) for t in batch[1:]]
            
            if waveforms.dim() == 3 and waveforms.shape[1] > 1:
                waveforms = torch.mean(waveforms, dim=1, keepdim=True)
            if waveforms.dim() == 3:
                waveforms = waveforms.squeeze(1)
                    
            spectrograms = preprocessor(waveforms, orig_sr=preprocessor.target_sr, augment=augment)
            
            if spectrograms.dim() == 3:
                spectrograms = spectrograms.unsqueeze(1)
                
            min_time = spectrograms.shape[-1]
            for i in range(len(targets)):
                if targets[i].dim() == 2:
                    targets[i] = targets[i].unsqueeze(1)
                min_time = min(min_time, targets[i].shape[-1])
                
            spectrograms = spectrograms[..., :min_time]
            targets = [t[..., :min_time] for t in targets]

            optimizer.zero_grad()
            
            amp_context = autocast("cuda") if use_amp and device=="cuda" else autocast("cpu")
                        
            with amp_context:
                outputs = model(spectrograms)
                
                # Dynamically handle model output type
                if isinstance(outputs, tuple):
                    # Dual Output Model (Assuming returns: onset_logits, frame_logits)
                    # And assuming dataloader yields: (waveforms, frame_labels, onset_labels)
                    onset_logits, frame_logits = outputs
                    frame_labels, onset_labels = targets[0], targets[1]
                    
                    onset_loss = criterion(onset_logits, onset_labels)
                    frame_loss = criterion(frame_logits, frame_labels)
                    loss = onset_loss + frame_loss
                else:
                    # Single Output Model
                    labels = targets[0]
                    loss = criterion(outputs, labels)

            # Backward pass 
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_train_loss += loss.item()
            
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                        
        avg_train_loss = running_train_loss / max(1, len(train_loader))
        
        if prune_per_epoch > 0 and epoch < epochs - 1:
            prune_model_step(model, prune_amount=prune_per_epoch)
            print(f"Pruned {prune_per_epoch*100}%")
        
        # --- VALIDATION PHASE ---
        print("starting validation")
        val_metrics = eval_metrics(model, val_loader, preprocessor)
        
        avg_val_loss = val_metrics.get("Loss", float("inf"))
        val_f1 = val_metrics.get("F1_Score", 0.0)

        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1}%")

        # Save weights if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved. Weights saved to {model_path}")

    print("\nTraining Complete!")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    #post training compression
    if prune_per_epoch > 0.0:
        make_pruning_permanent(model)
        
    if quantize:
        print("\n Starting quantization to INT8")
        model.eval()
        model.to("cpu")
        
        model = torch.quantization.quantize_dynamic(model, {nn.GRU, nn.Linear}, dtype=torch.qint8)
        quant_path = str(model_path).replace(".pth", "_quantized.pth")
        torch.save(model.state_dict(), quant_path)
        print(f"quantized weights saved to {quant_path}")
        
    return model