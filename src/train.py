import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-4, save_dir="trained_models"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Initializing model on {device}...")
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load data
    best_val_loss = float('inf')
    
    save_dir = Path(save_dir)
    model_name = model.__class__.__name__
    save_dir.mkdir(parents=True, exist_ok=True)
    model_iteration = len(list(save_dir.glob(f"{model_name}*")))
    model_path = save_dir / f"{model_name}_v{model_iteration+1}.pth"

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, (spectrograms, labels) in enumerate(train_loader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(spectrograms)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx} | Train Loss: {loss.item():.4f}")

        avg_train_loss = running_train_loss / max(1, len(train_loader))
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)
                
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / max(1, len(val_loader))
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # save weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  -> Validation loss improved. Weights saved to {model_path}")

    print("\nTraining Complete!")