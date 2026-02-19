from pathlib import Path
import csv
from dataclasses import dataclass
from torchaudio import info

# check info on MAESTRO database

@dataclass
class MaestroMetadata:
    composer: str
    title: str
    split: str
    year: int
    midi_filename: str
    midi_path: Path
    audio_filename: str
    audio_path: Path
    duration: float



def load_metadata(maestro_path: str) -> list[MaestroMetadata]:
    dataset = []
    
    csv_path = Path(maestro_path) / "maestro-v3.0.0.csv"
    
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            item = MaestroMetadata(
                composer = row["canonical_composer"],
                title = row["canonical_title"],
                split = row["split"],
                year = int(row["year"]),
                
                midi_filename = row["midi_filename"],
                midi_path = Path(maestro_path) / row["midi_filename"], # resolved path
                audio_filename = row["audio_filename"],
                audio_path = Path(maestro_path) / row["audio_filename"],
                
                duration = float(row["duration"])
            )
            dataset.append(item)
    
    return dataset