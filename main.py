from src.preprocess import load_metadata, MaestroPreprocessor
from src.dataset import MaestroDataset
from src.plot import plot_debug_cqts
import torch
import torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

maestro_path = r"D:/databases/maestro-v3.0.0"
ds = load_metadata(maestro_path)

maestro = MaestroPreprocessor(target_sr=22222).to(device)

# preprocess audio
for i, song in enumerate(ds[:1]):

    waveform, orig_sr = torchaudio.load(song.audio_path)
    waveform = waveform.to(device)
    
    x = maestro(waveform, orig_sr, augment=True, debug=True)
    
    plot_debug_cqts(*x)
        
    print(f"done {i}")

MaestroDataset(maestro_path, maestro)