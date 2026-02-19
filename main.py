from src.preprocess.helper import load_metadata
from pathlib import Path

maestro_path = r"D:/databases/maestro-v3.0.0"
ds = load_metadata(maestro_path)

song_1 = ds[0]


(song_1.audio_path)


