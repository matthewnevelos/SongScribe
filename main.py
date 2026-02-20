from src.preprocess import load_metadata, MaestroPipeline
from pathlib import Path
import torch
import torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

maestro_path = r"D:/databases/maestro-v3.0.0"
ds = load_metadata(maestro_path)

maestro = MaestroPipeline().to(device)

waveform, orig_sr = torchaudio.load(ds[0].audio_path)
waveform = waveform.to(device)

cqt_image = maestro(waveform, orig_sr)