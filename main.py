from src.preprocess import load_metadata, MaestroPreprocessor
from src.plot import plot_debug_cqts
from pathlib import Path
import torch
import torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

maestro_path = r"D:/databases/maestro-v3.0.0"
ds = load_metadata(maestro_path)

maestro = MaestroPreprocessor(chunk_sparsity=0.5, time_chunk_size=(20,100), freq_chunk_size=(1, 10), time_sparsity=0.5).to(device)

# preprocess audio
for i, song in enumerate(ds[:1]):

    waveform, orig_sr = torchaudio.load(song.audio_path)
    waveform = waveform.to(device)
    
    x, y = maestro(waveform, orig_sr, midi_path=song.midi_path, augment=True, debug=True)
    y = y.to(device)
    
    plot_debug_cqts(*x)
    
    print(f"{x.shape=}")
    print(f"{y.shape=}")
    
    print(f"done {i}")
