from src.preprocess import load_metadata
from pathlib import Path
import torch
import torchaudio


maestro_path = r"D:/databases/maestro-v3.0.0"
ds = load_metadata(maestro_path)

# print(torchaudio.info(ds[0].audio_path).sample_rate)

for song in ds:
    sr = torchaudio.info(song.audio_path).sample_rate
    if sr not in [48000, 44100]:
        print(sr)



# print(a.device)