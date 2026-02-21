from src.preprocess import load_metadata, MaestroPreprocessor
from pathlib import Path
import torch
import torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

maestro_path = r"D:/databases/maestro-v3.0.0"
ds = load_metadata(maestro_path)

maestro = MaestroPreprocessor().to(device)

# preprocess audio
for i, song in enumerate(ds[:4]):

    waveform, orig_sr = torchaudio.load(song.audio_path)
    waveform = waveform.to(device)
    
    x, y = maestro(waveform, orig_sr, midi_path=song.midi_path, augment=True)
    y  =y.to(device)
    
    print(f"{x.shape=}")
    print(f"{y.shape=}")
    
    print(f"done {i}")
