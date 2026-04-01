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
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg

from models import PianoTranscriptionCRNN
from preprocess import MaestroPreprocessor

def estimate_tempo(audio_path):
    y, sr = librosa.load(audio_path)
    # onset_env tracks the 'beats' or energy spikes in the audio
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    return int(round(float(tempo[0])))  # Round to the nearest whole number (integer)

def tensor_to_midi(onset_probs, frame_probs, frames_per_second, output_filepath):
    # Thresholds
    onset_threshold = 0.5
    frame_threshold = 0.3 
    
    # Create boolean arrays
    onsets = (onset_probs > onset_threshold)
    frames = (frame_probs > frame_threshold)
    
    pm = pretty_midi.PrettyMIDI()
    
    # Create TWO separate tracks for Right Hand (Treble) and Left Hand (Bass)
    right_hand = pretty_midi.Instrument(program=0, name="Treble")
    left_hand = pretty_midi.Instrument(program=0, name="Bass")
    
    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21 # A0 is MIDI note 21
        
        is_playing = False
        start_time = 0.0
        
        for t in range(onsets.shape[1]):
            # State 1: A new note is struck
            if onsets[pitch_idx, t] and not is_playing:
                start_time = t / frames_per_second
                is_playing = True
                
            # State 2: A note was struck AGAIN while the previous one was still sustaining
            elif onsets[pitch_idx, t] and is_playing:
                end_time = t / frames_per_second
                note = pretty_midi.Note(velocity=80, pitch=midi_pitch, start=start_time, end=end_time)
                
                # Split hands at Middle C (Pitch 60)
                if midi_pitch >= 60:
                    right_hand.notes.append(note)
                else:
                    left_hand.notes.append(note)
                    
                start_time = end_time # Immediately start the new note
                
            # State 3: The sustain dies out
            elif not frames[pitch_idx, t] and is_playing:
                end_time = t / frames_per_second
                
                # Ensure minimum length (50ms) to prevent music21 validation crashes
                if end_time - start_time > 0.05: 
                    note = pretty_midi.Note(velocity=80, pitch=midi_pitch, start=start_time, end=end_time)
                    if midi_pitch >= 60:
                        right_hand.notes.append(note)
                    else:
                        left_hand.notes.append(note)
                        
                is_playing = False
                
        # State 4: Catch any notes still playing at the very end of the audio file
        if is_playing:
            end_time = onsets.shape[1] / frames_per_second
            if end_time - start_time > 0.05:
                note = pretty_midi.Note(velocity=80, pitch=midi_pitch, start=start_time, end=end_time)
                if midi_pitch >= 60:
                    right_hand.notes.append(note)
                else:
                    left_hand.notes.append(note)

    # Only add the staves to the MIDI file if they actually played notes
    # (Prevents fatal empty <part> tags in MusicXML)
    if len(right_hand.notes) > 0:
        pm.instruments.append(right_hand)
    if len(left_hand.notes) > 0:
        pm.instruments.append(left_hand)
        
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

    spec_path = output_dir / f"{Path(audio_path).stem}_spectrogram.png"
    plt.figure(figsize=(16, 6))
    # Using 'magma' colormap as it clearly highlights high-energy frequencies
    plt.imshow(full_spectrogram.numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.title(f"CQT Spectrogram: {Path(audio_path).name}")
    plt.ylabel("MIDI Pitch (Index 0 = A0)")
    plt.xlabel("Time (Frames)")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.savefig(spec_path, dpi=150)
    plt.close()
    print(f"Saved Spectrogram to: {spec_path}")
    
    # 4. Convert to MIDI
    midi_path = output_dir / f"{Path(audio_path).stem}_transcribed.mid"
    
    # Call the updated dual-head tensor_to_midi function
    tensor_to_midi(onset_preds, frame_preds, frames_per_second, midi_path)
    
    # 5. Convert to Sheet Music
    print("Converting to Sheet Music...")
    try:
        score = music21.converter.parse(midi_path)
        
        mm = music21.tempo.MetronomeMark(number=tempo)
        ts = music21.meter.TimeSignature('4/4')
        
        # FIX: Inject metadata into the individual Parts (Staves), not the master Score
        if score.parts:
            for part in score.parts:
                part.insert(0, mm)
                part.insert(0, ts)
        else:
            # Fallback if no parts exist (e.g., a single flat stream)
            score.insert(0, mm)
            score.insert(0, ts)
        
        # Quantize (Note: [1,2,4,8,16] allows down to 64th notes)
        score.quantize([1, 2, 4, 8, 16], inPlace=True)
        
        # FIX: Use the .musicxml extension for better software compatibility
        xml_path = output_dir / f"{Path(audio_path).stem}_transcribed.musicxml"
        score.write('musicxml', fp=str(xml_path))
        print(f"Saved Sheet Music XML to: {xml_path}")
        
        view_sheet_music_inline(xml_path, output_dir)
        
    except Exception as e:
        print(f"Failed to render sheet music: {e}")

def view_sheet_music_inline(xml_path, output_dir):
    """Renders MusicXML to PNG in the background and displays it via Matplotlib."""
    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    
    # 1. Find the MuseScore executable
    system = platform.system()
    if system == "Windows":
        paths = [
            r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
            r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
        ]
        mscore_exe = next((p for p in paths if Path(p).exists()), None)
    elif system == "Darwin": 
        mscore_exe = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
        if not Path(mscore_exe).exists():
            mscore_exe = "/Applications/MuseScore 3.app/Contents/MacOS/mscore"
    else: 
        mscore_exe = "mscore" 

    if not mscore_exe or (system != "Linux" and not Path(mscore_exe).exists()):
        print("MuseScore not found. Cannot render inline images.")
        return

    # 2. Command MuseScore to render PNGs
    print("Rendering sheet music images...")
    # MuseScore will automatically append "-1", "-2" for multiple pages
    png_base_path = output_dir / xml_path.stem
    
    try:
        command = [mscore_exe, "-o", f"{png_base_path}.png", str(xml_path)]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 3. Find all the generated pages
        # MuseScore outputs them as "filename-1.png", "filename-2.png", etc.
        pages = sorted(glob.glob(f"{png_base_path}-*.png"))
        
        if not pages:
            print("Failed to generate PNGs.")
            return
            
        # 4. Display the pages natively in Python
        for i, page_path in enumerate(pages):
            img = mpimg.imread(page_path)
            
            # Create a tall figure that mimics a piece of paper
            plt.figure(figsize=(10, 14)) 
            plt.imshow(img)
            plt.axis('off') # Hide the graph axes
            plt.title(f"Sheet Music - Page {i+1}", fontsize=14)
            plt.tight_layout()
            
            print(f"Displaying Page {i+1}... (Close the window to continue)")
            plt.show() # This pauses the script until you close the image window
            
    except subprocess.CalledProcessError as e:
        print(f"Background rendering failed: {e}")

if __name__ == "__main__":
    #AUDIO_FILE = "test_set/Sheep/Sheep.wav"
    AUDIO_FILE = "test_set/signal flags/Signal Flags.wav"
    MODEL_WEIGHTS = "trained_models/onsets_1.pth"
    OUTPUT_FOLDER = "test_set/sheep"
    
    transcribe_audio(AUDIO_FILE, MODEL_WEIGHTS, OUTPUT_FOLDER)
