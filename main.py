from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset
from src.models import PianoTranscriptionCNN
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocessor = MaestroPreprocessor().to(device)

# maestro_path = r"D:/databases/maestro-v3.0.0"
# maestro_path = r"Example Data/MAESTRO"
maestro_path = r"T:/Uni/Final year/Digital Winter Sem/Final Project/maestro-v3.0.0/maestro-v3.0.0"

processed_dir = Path(maestro_path).parent / f"midi_{preprocessor.target_sr}"
metadata = load_metadata(maestro_path)
if not processed_dir.exists():
    preprocessor.precompute_midi_labels(processed_dir, sum(metadata.values(), []))

train_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["train"])
valid_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["validation"])
test_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["test"])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

valid_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

test_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

'''
if __name__ == "__main__":
    for batch_idx, (waveforms, labels) in enumerate(train_dataloader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            #preprocess
            spectrograms = preprocessor(waveforms, orig_sr=train_dataset.sample_rate, augment=True)
            
            #make labels same size
            print(spectrograms.shape[-1], labels.shape[-1])
            min_frames = min(spectrograms.shape[-1], labels.shape[-1])
            spectrograms = spectrograms[..., :min_frames]
            labels = labels[..., :min_frames]
'''
            
            
if __name__ == "__main__":
    # 1. Initialize Model, Loss, and Optimizer
    # Note: Adjust freq_bins if you altered the default nnAudio CQT parameters
    model = PianoTranscriptionCNN(freq_bins=88).to(device) 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    
    # 2. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (waveforms, labels) in enumerate(train_dataloader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            if waveforms.dim() == 3:
                waveforms = waveforms.squeeze(1)
            
            #preprocess
            spectrograms = preprocessor(waveforms, orig_sr=train_dataset.sample_rate, augment=True)
            
            # CNN expects a channel dimension: (Batch, 1, Freq, Time)
            if spectrograms.dim() == 3:
                spectrograms = spectrograms.unsqueeze(1)
            
            # Safety Check: Ensure Time dimension matches exactly between input and labels.
            # Due to hop_length math, they can sometimes be off by 1 frame.
            min_time = min(spectrograms.shape[-1], labels.shape[-1])
            spectrograms = spectrograms[:, :, :, :min_time]
            labels = labels[:, :, :min_time]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(spectrograms)
            
            # Calculate loss and backpropagate
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"piano_transcription_epoch_{epoch+1}.pth")