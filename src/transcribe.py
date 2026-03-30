import torch
import torchaudio
import numpy as np
import librosa
from scipy.signal import medfilt
from pathlib import Path
from .format_converter import output_to_midi, midi_to_sheet
from .format_converter import audio_to_CQT


def transcribe_audio(audio_path, trained_model, output_dir, chunk_seconds=5.0, sr = 11050, hop_length=256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Load Model
    trained_model = trained_model.to(device)
    trained_model.eval()
    
    frames_per_second = sr / hop_length
    
    # load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # Stereo to Mono
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        
    waveform = waveform.to(device)
    
    # process chunk by chunk
    chunk_samples = int(chunk_seconds * sr)
    all_predictions = []
    
    print(f"Transcribing {audio_path}...")
    with torch.no_grad():
        for start_sample in range(0, waveform.shape[1], chunk_samples):
            end_sample = start_sample + chunk_samples
            wave_chunk = waveform[:, start_sample:end_sample]
            
            if wave_chunk.shape[1] < chunk_samples:
                padding_needed = chunk_samples - wave_chunk.shape[1]
                wave_chunk = torch.nn.functional.pad(wave_chunk, (0, padding_needed))
            
            spectrogram = audio_to_CQT(wave_chunk, sample_rate=sr, hop_length=hop_length)
            spectrogram = spectrogram.unsqueeze(1)
            
            outputs = trained_model(spectrogram) 
            probs = torch.sigmoid(outputs)
            all_predictions.append(probs.squeeze(0))

    full_prediction = torch.cat(all_predictions, dim=1)
    
    original_frames = int((waveform.shape[1] / sr) * frames_per_second)
    full_prediction = full_prediction[:, :original_frames]
    
    # Smoothing and Post-Processing
    probs_numpy = full_prediction.cpu().numpy()
    for i in range(probs_numpy.shape[0]):
        probs_numpy[i, :] = medfilt(probs_numpy[i, :], kernel_size=5)
    
    # Convert to MIDI
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    midi_path = output_dir / f"{Path(audio_path).stem}_transcribed.mid"
    
    midi_obj = output_to_midi(
        raw_output=probs_numpy, 
        threshold=0.5, 
        sample_rate=sr, 
        hop_length=hop_length
    )
    midi_obj.write(str(midi_path))
    print(f"Saved MIDI to: {midi_path}")
    
    # Guess tempo 
    y = waveform.squeeze().cpu().numpy()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    estimated_bpm = int(tempo[0]) if isinstance(tempo, np.ndarray) else int(tempo)
    
    print("Converting to Sheet Music...")
    try:
        sheet = midi_to_sheet(midi_obj, estimated_bpm)
        sheet_path = output_dir / f"{Path(audio_path).stem}_sheet.xml"
        sheet.write("musicxml", fp=str(sheet_path))
    except Exception as e:
        print(f"Failed to render sheet music: {e}")