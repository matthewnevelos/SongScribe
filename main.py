from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset
import torch
from torch.utils.data import DataLoader
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocessor = MaestroPreprocessor().to(device)

# maestro_path = r"D:/databases/maestro-v3.0.0"
maestro_path = r"Example Data/MAESTRO"

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
