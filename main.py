from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset
from src.models import PianoTranscriptionCRNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from torch.utils.data import DataLoader
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True # Benchmarks different algorithms to train faster

preprocessor = MaestroPreprocessor().to(device)

# maestro_path = r"D:/databases/maestro-v3.0.0"
# maestro_path = r"Example Data/MAESTRO"
maestro_path = r"T:/Uni/Final year/Digital Winter Sem/Final Project/maestro-v3.0.0/maestro-v3.0.0_22050Hz"

processed_dir = Path(maestro_path).parent / f"midi_{preprocessor.target_sr}"
metadata = load_metadata(maestro_path)
if not processed_dir.exists():
    preprocessor.precompute_midi_labels(processed_dir, sum(metadata.values(), []))

def prune_model_step(model, prune_amount=0.1):
    """
    Applies L1 Unstructured Pruning to all Convolutional and Linear layers.
    Removes the lowest (prune_amount * 100)% of weights globally.
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            
    # We don't strictly prune the GRUs here because pruning recurrent 
    # hidden-to-hidden matrices can cause mathematical instability, 
    # and they will be heavily compressed by quantization anyway.
            
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )
    print(f"[*] Pruned an additional {prune_amount*100}% of Conv/Linear weights.")

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

train_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["train"])
valid_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["validation"])
test_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["test"])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True
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
    # We use 88 freq_bins to match the nnAudio CQT output
    model = PianoTranscriptionCRNN(freq_bins=88).to(device) 
    
    # BCEWithLogitsLoss is still the correct choice, but now we use it for both heads
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam or AdamW are standard here
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    scaler = torch.amp.GradScaler('cuda') # Added to make the model train faster, calculates on 16-bit floats instead of 32.

    epochs = 5
    
    # If the model crashes during training you can resume from the latest epoch
    #start_epoch = 3
    #checkpoint_path = f"onsets_frames_epoch_{start_epoch}.pth"
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    prune_amount_per_epoch = 0.1 # Prune 10%

    print("Starting Training for Onsets and Frames CRNN...")
    
    # 2. Training Loop
    for epoch in range(epochs):
        # Change to range(start_epoch, epochs) if using checkpoint
        model.train()
        total_loss = 0.0
        total_onset_loss = 0.0
        total_frame_loss = 0.0
        
        # NOTE: Unpacking 3 variables now!
        for batch_idx, (waveforms, frame_labels, onset_labels) in enumerate(train_dataloader):
            waveforms = waveforms.to(device)
            frame_labels = frame_labels.to(device)
            onset_labels = onset_labels.to(device)
            
            # 1. Fix waveform shape for nnAudio [Batch, Frames]
            if waveforms.dim() == 3:
                waveforms = waveforms.squeeze(1) 
            
            # Preprocess: (Batch, Time) -> (Batch, Freq, Time)
            spectrograms = preprocessor(waveforms, orig_sr=train_dataset.sample_rate, augment=True)
            
            # 2. CNN expects a channel dimension: (Batch, 1, Freq, Time)
            if spectrograms.dim() == 3:
                spectrograms = spectrograms.unsqueeze(1)
            if frame_labels.dim() == 2:
                 frame_labels = frame_labels.unsqueeze(1)
            if onset_labels.dim() == 2:
                 onset_labels = onset_labels.unsqueeze(1)
            
            # Safety Check: Ensure Time dimension matches exactly
            min_time = min(spectrograms.shape[-1], frame_labels.shape[-1], onset_labels.shape[-1])
            spectrograms = spectrograms[:, :, :, :min_time]
            frame_labels = frame_labels[:, :, :min_time]
            onset_labels = onset_labels[:, :, :min_time]
            
            optimizer.zero_grad()
            
            # Forward pass calculated based on 16-bit floats
            with torch.amp.autocast('cuda'):
                onset_logits, frame_logits = model(spectrograms)
                onset_loss = criterion(onset_logits, onset_labels)
                frame_loss = criterion(frame_logits, frame_labels)
                loss = onset_loss + frame_loss
            
            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Tracking
            total_loss += loss.item()
            total_onset_loss += onset_loss.item()
            total_frame_loss += frame_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_dataloader)} "
                      f"| Total Loss: {loss.item():.4f} "
                      f"(Onset: {onset_loss.item():.4f}, Frame: {frame_loss.item():.4f})")
                
        # Epoch Summary
        avg_loss = total_loss / len(train_dataloader)
        avg_onset = total_onset_loss / len(train_dataloader)
        avg_frame = total_frame_loss / len(train_dataloader)

        if epoch < epochs - 1:
            prune_model_step(model, prune_amount=prune_amount_per_epoch)
        
        print(f"--- Epoch {epoch+1} Completed ---")
        print(f"Average Total Loss: {avg_loss:.4f} | Onset: {avg_onset:.4f} | Frame: {avg_frame:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"onsets_frames_epoch_{epoch+1}.pth")

    print("\nStarting Post-Training Compression...")
    
    # 1. Make Pruning Permanent
    make_pruning_permanent(model)
    
    # Move model to CPU for Dynamic Quantization (PyTorch requirement)
    model.eval()
    model.to('cpu')
    
    # 2. Dynamic Quantization
    # We target the GRU and Linear layers, converting their weights to int8.
    print("[*] Quantizing GRU and Linear layers to INT8...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.GRU, nn.Linear}, 
        dtype=torch.qint8
    )
    
    # 3. Save the final compressed model
    compressed_path = "onsets_frames_COMPRESSED_final.pth"
    torch.save(quantized_model.state_dict(), compressed_path)