import torch
import torchaudio
import numpy as np
import librosa
from scipy.signal import medfilt
from pathlib import Path

# Import your modular components
from format_converter import output_to_midi, midi_to_sheet
from models.crnn import PianoTranscriptionCRNN
from preprocess import MaestroPreprocessor

def transcribe_audio(audio_path, model_path, output_dir, chunk_seconds=5.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model and Preprocessor
    model = PianoTranscriptionCRNN(freq_bins=88).to(device)
    # Added weights_only=True to resolve the PyTorch security warning
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    preprocessor = MaestroPreprocessor().to(device)
    target_sr = preprocessor.target_sr
    frames_per_second = target_sr / preprocessor.hop_length
    
    # 2. Load Audio
    waveform, orig_sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # Stereo to Mono
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
        
    waveform = waveform.to(device)
    
    # 3. Process in Chunks (to save VRAM) and Concatenate
    chunk_samples = int(chunk_seconds * target_sr)
    all_predictions = []
    
    print(f"Transcribing {audio_path}...")
    with torch.no_grad():
        for start_sample in range(0, waveform.shape[1], chunk_samples):
            end_sample = start_sample + chunk_samples
            wave_chunk = waveform[:, start_sample:end_sample]
            
            if wave_chunk.shape[1] < chunk_samples:
                padding_needed = chunk_samples - wave_chunk.shape[1]
                wave_chunk = torch.nn.functional.pad(wave_chunk, (0, padding_needed))
            
            spectrogram = preprocessor(wave_chunk, target_sr, augment=False)
            spectrogram = spectrogram.unsqueeze(1)
            
            outputs = model(spectrogram) 
            probs = torch.sigmoid(outputs)
            all_predictions.append(probs.squeeze(0))

    full_prediction = torch.cat(all_predictions, dim=1)
    
    original_frames = int((waveform.shape[1] / target_sr) * frames_per_second)
    full_prediction = full_prediction[:, :original_frames]
    
    # 4. Smoothing and Post-Processing
    # Move to CPU and apply median filter to probabilities
    probs_numpy = full_prediction.cpu().numpy()
    for i in range(probs_numpy.shape[0]):
        probs_numpy[i, :] = medfilt(probs_numpy[i, :], kernel_size=5)
    
    # 5. Convert to MIDI
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    midi_path = output_dir / f"{Path(audio_path).stem}_transcribed.mid"
    
    # Call the new modular function
    midi_obj = output_to_midi(
        raw_output=probs_numpy, 
        threshold=0.5, 
        sample_rate=target_sr, 
        hop_length=preprocessor.hop_length
    )
    midi_obj.write(str(midi_path))
    print(f"Saved MIDI to: {midi_path}")
    
    # 6. Automate Tempo Detection 
    print("Estimating tempo from audio...")
    y = waveform.squeeze().cpu().numpy()
    tempo, _ = librosa.beat.beat_track(y=y, sr=target_sr)
    estimated_bpm = int(tempo[0]) if isinstance(tempo, np.ndarray) else int(tempo)
    print(f"Detected Tempo: {estimated_bpm} BPM")
    
    # 7. Convert to Sheet Music
    print("Converting to Sheet Music...")
    try:
        # Pass the memory object directly to music21
        midi_to_sheet(midi_obj, estimated_bpm)
    except Exception as e:
        print(f"Failed to render sheet music: {e}")

if __name__ == "__main__":
    AUDIO_FILE = "test_set/sheep/Sheep.wav"
    MODEL_WEIGHTS = "trained_models/crnn_1.pth"
    OUTPUT_FOLDER = "test_set/sheep"
    
    transcribe_audio(AUDIO_FILE, MODEL_WEIGHTS, OUTPUT_FOLDER)