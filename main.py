from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset
from src.models import *
from src.plot import *
from src.train import train_model
from src.transcribe import transcribe_audio
from src.export_onnx import export_onnx
import torch
from torch.utils.data import DataLoader
            
if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = MaestroPreprocessor(hop_length=128).to(device)

    # defines the path of csv and raw data
    maestro_path = r"D:/databases/maestro-v3.0.0"
    metadata = load_metadata(maestro_path)

    #precompute audio to Pytorch tensors
    preprocessor.precompute_full_dataset(maestro_path)

    # set up datasets
    train_dataset = MaestroDataset(
        maestro_dir=maestro_path, 
        target_sr=preprocessor.target_sr, 
        hop_length=preprocessor.hop_length, 
        metadata=metadata["train"],
        segment_seconds=5,
        return_onsets=True)

    valid_dataset = MaestroDataset(
        maestro_dir=maestro_path, 
        target_sr=preprocessor.target_sr, 
        hop_length=preprocessor.hop_length, 
        metadata=metadata["validation"],
        segment_seconds=5,
        return_onsets=True)

    test_dataset = MaestroDataset(
        maestro_dir=maestro_path, 
        target_sr=preprocessor.target_sr, 
        hop_length=preprocessor.hop_length, 
        metadata=metadata["test"],
        segment_seconds=5,
        return_onsets=True)
    
    #Set up data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    model = PianoOnsetFrameCRNN().to(device)

    #training loop
    model = train_model(
        model,
        preprocessor, 
        train_dataloader,
        valid_dataloader,
        epochs=3,
        augment=False)
    
    #load model if just evaluating
    state_dict = torch.load("trained_models/onsets_frames_epoch_5.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    #convert to onnx model for interactive GitHub Pages app
    export_onnx(model, "docs/songscribe.onnx")
    
    #transcribe baa baa black sheep
    AUDIO_FILE = "test_set/sheep/Sheep.wav"
    OUTPUT_FOLDER = "test_set/sheep"
    sheet = transcribe_audio(AUDIO_FILE, model, OUTPUT_FOLDER, onset_threshold=0.3,)      

    # transcribe signal flags
    AUDIO_FILE = "test_set/signal flags/Signal Flags.wav"
    OUTPUT_FOLDER = "test_set/signal flags"
    sheet = transcribe_audio(AUDIO_FILE, model, OUTPUT_FOLDER, show=True)