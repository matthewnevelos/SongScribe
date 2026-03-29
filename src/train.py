import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

def train_model(model, preprocessor, train_loader, val_loader, epochs=5, lr=1e-4, save_dir="trained_models"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Initializing model on {device}...")
    model = model.to(device)
    preprocessor = preprocessor.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    save_dir = Path(save_dir)
    model_name = model.__class__.__name__
    save_dir.mkdir(parents=True, exist_ok=True)
    model_iteration = len(list(save_dir.glob(f"{model_name}*")))
    model_path = save_dir / f"{model_name}_v{model_iteration+1}.pth"
    print(f"Weights will be saved to: {model_path}")

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (waveforms, labels) in enumerate(train_bar):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            if waveforms.dim() == 3:
                if waveforms.shape[0] == 1 and waveforms.shape[1] > 1:
                    waveforms = waveforms.squeeze(0)
                else:
                    waveforms = waveforms.squeeze(1)
                    
            spectrograms = preprocessor(waveforms, orig_sr=preprocessor.target_sr, augment=True)
            
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
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                
                if waveforms.dim() == 3:
                    if waveforms.shape[0] == 1 and waveforms.shape[1] > 1:
                        waveforms = waveforms.squeeze(0)
                    else:
                        waveforms = waveforms.squeeze(1)
                
                spectrograms = preprocessor(waveforms, orig_sr=preprocessor.target_sr, augment=False)
                
                if spectrograms.dim() == 3:
                    spectrograms = spectrograms.unsqueeze(1)
                    
                min_time = min(spectrograms.shape[-1], labels.shape[-1])
                spectrograms = spectrograms[:, :, :, :min_time]
                labels = labels[:, :, :min_time]
                
                # Evaluate
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / max(1, len(val_loader))
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save weights if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved. Weights saved to {model_path}")

    print("\nTraining Complete!")