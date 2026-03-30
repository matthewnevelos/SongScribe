import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from .evaluate import evaluate_model

def train_model(model, preprocessor, train_loader, val_loader, epochs=5, lr=1e-4, save_dir="trained_models", model_name=None, augment=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Initializing model on {device}...")
    model = model.to(device)
    preprocessor = preprocessor.to(device)
    
    positive_weight = torch.tensor([15.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        
        for batch_idx, (waveforms, labels) in enumerate(train_bar):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            if waveforms.dim() == 3:
                if waveforms.shape[0] == 1 and waveforms.shape[1] > 1:
                    waveforms = waveforms.squeeze(0)
                else:
                    waveforms = waveforms.squeeze(1)
                    
            spectrograms = preprocessor(waveforms, orig_sr=preprocessor.target_sr, augment=augment)
            
            if spectrograms.dim() == 3:
                spectrograms = spectrograms.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                        
        avg_train_loss = running_train_loss / max(1, len(train_loader))
        
        # --- VALIDATION PHASE ---
        print("starting validation")
        val_metrics = evaluate_model(model, val_loader)
        avg_val_loss = val_metrics["Loss"]
        val_f1 = val_metrics["F1_Score"]

        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1}%")

        # Save weights if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved. Weights saved to {model_path}")

    print("\nTraining Complete!")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model