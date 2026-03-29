from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset
from src.models.crnn import PianoTranscriptionCRNN
from src.train import train_model
from src.evaluate import evaluate_model
import torch
from torch.utils.data import DataLoader
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocessor = MaestroPreprocessor().to(device)

maestro_path = r"D:/databases/maestro-v3.0.0"
# maestro_path = r"Example Data/MAESTRO"
# maestro_path = r"T:/Uni/Final year/Digital Winter Sem/Final Project/maestro-v3.0.0/maestro-v3.0.0"

processed_dir = Path(maestro_path).parent / f"midi_{preprocessor.target_sr}"
metadata = load_metadata(maestro_path)
if not processed_dir.exists():
    preprocessor.precompute_midi_labels(processed_dir, sum(metadata.values(), []))

train_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["train"])
valid_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["validation"])
test_dataset = MaestroDataset(maestro_dir=maestro_path, target_sr=preprocessor.target_sr, hop_length=preprocessor.hop_length, metadata=metadata["test"])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

valid_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

test_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True
)
     
            
if __name__ == "__main__":
    model = PianoTranscriptionCRNN().to(device)
    
    train_model(
        model,
        preprocessor, 
        train_dataloader,
        valid_dataloader)