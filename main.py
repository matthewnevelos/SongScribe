from src.preprocess import MaestroPreprocessor, load_metadata
from src.dataset import MaestroDataset
from src.models import *
from src.plot import *
from src.train import train_model
from src.transcribe import transcribe_audio
from src.format_converter import midi_to_sheet, audio_to_CQT, audio_to_STFT
from src.evaluate import eval_metrics
import torch
from torch.utils.data import DataLoader
from pathlib import Path
            
if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = MaestroPreprocessor(hop_length=128).to(device)

    #defines the path of csv and raw data
    maestro_path = r"D:/databases/maestro-v3.0.0"
    # maestro_path = r"Example Data/MAESTRO"
    # maestro_path = r"T:/Uni/Final year/Digital Winter Sem/Final Project/maestro-v3.0.0/maestro-v3.0.0"

    preprocessor.precompute_full_dataset(maestro_path)
    metadata = load_metadata(maestro_path)

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

    model = OnsetCRNN().to(device)

    model = train_model(
        model,
        preprocessor, 
        train_dataloader,
        valid_dataloader,
        epochs=3,
        augment=False)
    
    # state_dict = torch.load("trained_models/onsets_frames_epoch_4.pth", map_location=device, weights_only=True)
    # model.load_state_dict(state_dict)
    
    AUDIO_FILE = "test_set/sheep/Sheep.wav"
    OUTPUT_FOLDER = "test_set/sheep"
    # audio, _ = torchaudio.load(AUDIO_FILE)
    # plot_CQT(preprocessor(audio, augment=False, orig_sr=preprocessor.target_sr), show=True)
    sheet = transcribe_audio(AUDIO_FILE, model, OUTPUT_FOLDER)
    
    AUDIO_FILE = "test_set/signal flags/Signal Flags.wav"
    OUTPUT_FOLDER = "test_set/signal flags"
    sheet = transcribe_audio(AUDIO_FILE, model, OUTPUT_FOLDER, frame_threshold=0.2, onset_threshold=0.9, show=True)
    
            
    # # metrics = eval_metrics(model, test_dataloader, threshold=0.5)
    
    # # AUDIO_FILE = "test_set/C/piano_c.ogg"
    # AUDIO_FILE = "test_set/sheep/Sheep.wav"
    # # AUDIO_FILE = "test_set/signal flags/Signal Flags.wav"
    # audio, sr = torchaudio.load(AUDIO_FILE)
    # cqt = audio_to_STFT(audio)
    # plot_CQT(cqt[0], sr, show=True)
