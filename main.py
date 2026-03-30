from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset
from src.models.unet import PianoUNet_v1
from src.models.crnn import PianoTranscriptionCRNN_v1
from src.train import train_model
from src.transcribe import transcribe_audio
from src.format_converter import midi_to_sheet
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
    drop_last=True,
)

valid_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

test_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
     
            
if __name__ == "__main__":
    model = PianoUNet_v1().to(device)
    
    # model = train_model(
    #     model,
    #     preprocessor, 
    #     train_dataloader,
    #     valid_dataloader,
    #     epochs=1)
    
    state_dict = torch.load("trained_models/PianoUNet_v1.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    AUDIO_FILE = "test_set/sheep/Sheep.wav"
    OUTPUT_FOLDER = "test_set/sheep"
    
    # AUDIO_FILE = "test_set/signal flags/Signal Flags.wav"
    # OUTPUT_FOLDER = "test_set/signal flags"
    
    sheet = transcribe_audio(AUDIO_FILE, model, OUTPUT_FOLDER)
            
    # metrics = evaluate_model(model, test_dataloader, threshold=0.5)