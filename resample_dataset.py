import os
import csv
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


ORIGINAL_MAESTRO_DIR = Path(r"T:/Uni/Final year/Digital Winter Sem/Final Project/maestro-v3.0.0/maestro-v3.0.0")

TARGET_SR = 22050

NEW_MAESTRO_DIR = Path(str(ORIGINAL_MAESTRO_DIR) + f"_{TARGET_SR}Hz")
# ---------------------

def process_single_file(args):
    """Worker function to resample a single audio file."""
    orig_audio_path, new_audio_path, target_sr = args
    
    # Skip if already processed
    if new_audio_path.exists():
        return True
        
    try:
        # Create subdirectories
        new_audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        waveform, sr = torchaudio.load(orig_audio_path)
        
        # 1. Convert Stereo to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 2. Resample to Target SR
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
        # Save optimized audio
        torchaudio.save(new_audio_path, waveform, target_sr)
        return True
        
    except Exception as e:
        return f"Error processing {orig_audio_path.name}: {e}"

if __name__ == '__main__':
    print(f"Original Directory: {ORIGINAL_MAESTRO_DIR}")
    print(f"Target Directory:   {NEW_MAESTRO_DIR}")
    print(f"Target Sample Rate: {TARGET_SR} Hz\n")
    
    NEW_MAESTRO_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = ORIGINAL_MAESTRO_DIR / "maestro-v3.0.0.csv"
    new_csv_path = NEW_MAESTRO_DIR / "maestro-v3.0.0.csv"
    
    # Copy the CSV to the new folder so it acts exactly like a normal MAESTRO dataset
    if not new_csv_path.exists():
        new_csv_path.write_text(csv_path.read_text(encoding='utf-8'), encoding='utf-8')
    
    # Build a list of all files that need processing
    tasks = []
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            orig_audio = ORIGINAL_MAESTRO_DIR / row['audio_filename']
            new_audio = NEW_MAESTRO_DIR / row['audio_filename']
            tasks.append((orig_audio, new_audio, TARGET_SR))
            
    print(f"Found {len(tasks)} audio files. Starting multiprocess resampling...")

    successful = 0
    with ProcessPoolExecutor() as executor:
        # Map tasks to the executor and wrap with tqdm for a progress bar
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resampling Audio"):
            result = future.result()
            if result is True:
                successful += 1
            else:
                print(result)
                
    print(f"\n Successfully processed {successful}/{len(tasks)} files.")