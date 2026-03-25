import torch
import torchaudio
import numpy as np
import pretty_midi
import music21
from scipy.signal import medfilt
from pathlib import Path

# Import your custom modules
from src.models import PianoTranscriptionCRNN
from src.preprocess import MaestroPreprocessor

def tensor_to_midi(prediction_tensor, frames_per_second, output_filepath):
    """
    Converts a (88, Time) binary prediction tensor into a MIDI file.
    """
    # prediction_tensor shape: (88, Time)
    binary_predictions = (prediction_tensor > 0.5).int().cpu().numpy()

    # Apply a median filter across the time axis (kernel size 5 or 7 frames works well)
    for i in range(binary_predictions.shape[0]):
        binary_predictions[i, :] = medfilt(binary_predictions[i, :], kernel_size=5)

    pm = pretty_midi.PrettyMIDI()
    
    # Create TWO separate tracks for Right Hand and Left Hand
    right_hand = pretty_midi.Instrument(program=0, name="Treble")
    left_hand = pretty_midi.Instrument(program=0, name="Bass")
    
    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21
        active_frames = binary_predictions[pitch_idx, :]
        
        # Find note starts and ends (transitions from 0 to 1, and 1 to 0)
        diff = np.diff(np.insert(np.insert(active_frames, 0, 0), len(active_frames), 0))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            start_time = start / frames_per_second
            end_time = end / frames_per_second
            
            note = pretty_midi.Note(velocity=100, pitch=midi_pitch, start=start_time, end=end_time)
        
            # Split hands at Middle C (Pitch 60)
            if midi_pitch >= 60:
                right_hand.notes.append(note)
            else:
                left_hand.notes.append(note)
                
    pm.instruments.append(right_hand)
    pm.instruments.append(left_hand)
    pm.write(str(output_filepath))

def transcribe_audio(audio_path, model_path, output_dir, chunk_seconds=5.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model and Preprocessor
    model = PianoTranscriptionCRNN(freq_bins=88).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
            
            # --- FIX: Pad the final tail chunk with zeros if it's too short ---
            if wave_chunk.shape[1] < chunk_samples:
                padding_needed = chunk_samples - wave_chunk.shape[1]
                wave_chunk = torch.nn.functional.pad(wave_chunk, (0, padding_needed))
            
            # Preprocess (CQT) - augment=False for inference!
            spectrogram = preprocessor(wave_chunk, target_sr, augment=False)
            
            # Add channel dimension for CNN: (Batch=1, Channel=1, Freq, Time)
            spectrogram = spectrogram.unsqueeze(1)
            
            # Forward pass
            outputs = model(spectrogram) 
            
            # Apply Sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(outputs)

            all_predictions.append(probs.squeeze(0)) # Remove batch dim

    # Stitch the time dimension together
    full_prediction = torch.cat(all_predictions, dim=1)
    
    # --- FIX: Trim the extra padded silence off the final prediction ---
    original_frames = int((waveform.shape[1] / target_sr) * frames_per_second)
    full_prediction = full_prediction[:, :original_frames]
    
    # 4. Convert to MIDI
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    midi_path = output_dir / f"{Path(audio_path).stem}_transcribed.mid"
    
    tensor_to_midi(full_prediction, frames_per_second, midi_path)
    
    # 5. Convert to Sheet Music using Music21
    print("Converting to Sheet Music...")
    try:
        # Parse the MIDI file
        score = music21.converter.parse(midi_path)
        
        # 1. Inject the Tempo (Estimate what you played, e.g., 100 BPM)
        mm = music21.tempo.MetronomeMark(number=130)
        score.insert(0, mm)
        
        # 2. Inject the Time Signature (Baa Baa Black Sheep is in 4/4)
        ts = music21.meter.TimeSignature('4/4')
        score.insert(0, ts)
        
        # 3. NOW Quantize, and it will snap to the correct 4/4 grid at 100 BPM
        score.quantize([1, 2, 4, 8, 16], inPlace=True)
        
        # Save as MusicXML (standard sheet music format)
        xml_path = output_dir / f"{Path(audio_path).stem}_sheet.xml"
        score.write('musicxml', fp=str(xml_path))
        print(f"Saved Sheet Music XML to: {xml_path}")
        
        # Opens your default sheet music viewer (e.g., MuseScore)
        score.show() 
    except Exception as e:
        print(f"Failed to render sheet music: {e}")

if __name__ == "__main__":
    AUDIO_FILE = "Sheep.wav"
    MODEL_WEIGHTS = "piano-model-v2.pth"
    OUTPUT_FOLDER = "transcriptions"
    
    transcribe_audio(AUDIO_FILE, MODEL_WEIGHTS, OUTPUT_FOLDER)