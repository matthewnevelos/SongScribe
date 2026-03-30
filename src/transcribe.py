import torchaudio
try:   ### ADDED
    torchaudio.set_audio_backend("soundfile")
except:
    pass 

import torch
import numpy as np
import pretty_midi
import music21
from scipy.signal import medfilt
from pathlib import Path
import librosa

from models import PianoTranscriptionCRNN
from preprocess import MaestroPreprocessor

### ADDED
def estimate_tempo(audio_path):
    y, sr = librosa.load(audio_path)
    # onset_env tracks the 'beats' or energy spikes in the audio
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    return int(round(float(tempo[0])))  # Round to the nearest whole number (integer)

def tensor_to_midi(onset_probs, frame_probs, frames_per_second, output_filepath):
    # Thresholds
    onset_threshold = 0.5
    frame_threshold = 0.3 # Frame threshold can be lower to catch the fading tail
    
    # Convert to binary
    onsets = (onset_probs > onset_threshold)
    frames = (frame_probs > frame_threshold)
    
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21
        
        is_playing = False
        start_time = 0.0
        
        for t in range(onsets.shape[1]):
            # State 1: A new note is struck
            if onsets[pitch_idx, t] and not is_playing:
                start_time = t / frames_per_second
                is_playing = True
                
            # State 2: A note was struck AGAIN while the previous one was still sustaining
            elif onsets[pitch_idx, t] and is_playing:
                # End the previous note right here
                end_time = t / frames_per_second
                piano.notes.append(pretty_midi.Note(80, midi_pitch, start_time, end_time))
                # Immediately start the new note
                start_time = end_time 
                
            # State 3: The sustain dies out
            elif not frames[pitch_idx, t] and is_playing:
                end_time = t / frames_per_second
                # Ensure the note has a minimum length so music21 doesn't crash
                if end_time - start_time > 0.05: 
                    piano.notes.append(pretty_midi.Note(80, midi_pitch, start_time, end_time))
                is_playing = False
                
        # Catch any notes still playing at the very end of the song
        if is_playing:
            end_time = onsets.shape[1] / frames_per_second
            piano.notes.append(pretty_midi.Note(80, midi_pitch, start_time, end_time))

    pm.instruments.append(piano)
    pm.write(str(output_filepath))

def transcribe_audio(audio_path, model_path, output_dir, chunk_seconds=5.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model and Preprocessor
    model = PianoTranscriptionCRNN(freq_bins=88).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    preprocessor = MaestroPreprocessor().to(device)
    target_sr = preprocessor.target_sr
    frames_per_second = target_sr / preprocessor.hop_length
    
    # 2. Load Audio, Trim, and Get BPM (Keep your existing librosa logic here)
    print(f"Loading {audio_path}...")
    waveform, orig_sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
        
    waveform_np = waveform.squeeze().numpy()
    trimmed_np, index = librosa.effects.trim(waveform_np, top_db=30)
    start_sample = index[0]
    waveform = waveform[:, start_sample:]
    print(f"Trimmed {start_sample / target_sr:.2f} seconds of silence.")
    
    tempo, _ = librosa.beat.beat_track(y=trimmed_np, sr=target_sr)
    estimated_bpm = int(round(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)))
    print(f"Estimated Tempo: {estimated_bpm} BPM")
    
    waveform = waveform.to(device)
    
    # 3. Process in Chunks
    chunk_samples = int(chunk_seconds * target_sr)
    
    # Track onsets, frames, and spectrograms separately
    all_onsets = []
    all_frames = []
    all_spectrograms = [] 
    
    print("Transcribing audio...")
    with torch.no_grad():
        for start_idx in range(0, waveform.shape[1], chunk_samples):
            end_idx = start_idx + chunk_samples
            wave_chunk = waveform[:, start_idx:end_idx]
            
            if wave_chunk.shape[1] < chunk_samples:
                padding_needed = chunk_samples - wave_chunk.shape[1]
                wave_chunk = torch.nn.functional.pad(wave_chunk, (0, padding_needed))
            
            spectrogram = preprocessor(wave_chunk, orig_sr=target_sr, augment=False)
            
            # Save the raw 2D spectrogram for plotting before adding the channel dimension
            all_spectrograms.append(spectrogram.cpu().squeeze(0))
            
            spectrogram = spectrogram.unsqueeze(1)
            
            # --- FIX: Unpack the dual-head outputs ---
            onset_logits, frame_logits = model(spectrogram) 
            
            onset_probs = torch.sigmoid(onset_logits)
            frame_probs = torch.sigmoid(frame_logits)
            
            all_onsets.append(onset_probs.squeeze(0))
            all_frames.append(frame_probs.squeeze(0))

    # Stitch everything together
    full_onsets = torch.cat(all_onsets, dim=1)
    full_frames = torch.cat(all_frames, dim=1)
    full_spectrogram = torch.cat(all_spectrograms, dim=1)
    
    # Trim the extra padded silence off the ends
    original_frames = int((waveform.shape[1] / target_sr) * frames_per_second)
    full_onsets = full_onsets[:, :original_frames]
    full_frames = full_frames[:, :original_frames]
    full_spectrogram = full_spectrogram[:, :original_frames]
    
    # Convert to numpy for filtering and MIDI generation
    onset_preds = full_onsets.cpu().numpy()
    frame_preds = full_frames.cpu().numpy()
    
    # Median Filter: Apply ONLY to frames. Onsets are too short and would be erased.
    for i in range(frame_preds.shape[0]):
        frame_preds[i, :] = medfilt(frame_preds[i, :], kernel_size=5)
    
    # --- NEW: Save Spectrogram Image ---
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 4. Convert to MIDI
    midi_path = output_dir / f"{Path(audio_path).stem}_transcribed.mid"
    
    # Call the updated dual-head tensor_to_midi function
    tensor_to_midi(onset_preds, frame_preds, frames_per_second, midi_path)
    
    # 5. Convert to Sheet Music
    print("Converting to Sheet Music...")
    try:
        score = music21.converter.parse(midi_path)
        mm = music21.tempo.MetronomeMark(number=estimated_bpm)
        score.insert(0, mm)
        ts = music21.meter.TimeSignature('4/4')
        score.insert(0, ts)
        
        score.quantize([4, 8, 16], inPlace=True)
        
        xml_path = output_dir / f"{Path(audio_path).stem}_sheet.xml"
        score.write('musicxml', fp=str(xml_path))
        print(f"Saved Sheet Music XML to: {xml_path}")
        score.show() 
    except Exception as e:
        print(f"Failed to render sheet music: {e}")

if __name__ == "__main__":
    AUDIO_FILE = "test_set/sheep/Sheep.wav"
    MODEL_WEIGHTS = "trained_models/onsets_1.pth"
    OUTPUT_FOLDER = "test_set/sheep"
    
    transcribe_audio(AUDIO_FILE, MODEL_WEIGHTS, OUTPUT_FOLDER)
