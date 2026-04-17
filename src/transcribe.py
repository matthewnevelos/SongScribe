import torch
import torchaudio
import numpy as np
import librosa
from scipy.signal import medfilt
from pathlib import Path
from .format_converter import output_to_midi, midi_to_sheet, audio_to_CQT
from .evaluate import binarize_output, eval_metrics


def transcribe_audio(audio_path, trained_model, output_dir, chunk_seconds=5.0, sr = 22050, hop_length=256,
                     frame_threshold = 0.3, onset_threshold=0.5, show=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Load Model
    trained_model = trained_model.to(device)
    trained_model.eval()
    
    # load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # Stereo to Mono
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        
    waveform = waveform.to(device)
    
    # process chunk by chunk
    chunk_samples = int(chunk_seconds * sr)
    all_onset_probs = [] # single output models will ignore this
    all_frame_probs = []
    
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
            
            if isinstance(outputs, tuple):
                onset_probs = torch.sigmoid(outputs[0]).squeeze(0)
                frame_probs = torch.sigmoid(outputs[1]).squeeze(0)
                all_onset_probs.append(onset_probs)
                all_frame_probs.append(frame_probs)
            else:
                # Standard single-output model
                probs = torch.sigmoid(outputs).squeeze(0)
                all_frame_probs.append(probs)
                

    all_frame_probs = torch.cat(all_frame_probs, dim=1)
    original_frames = int(waveform.shape[1] // hop_length)
    all_frame_probs = all_frame_probs[:, :original_frames]
    
    # Reconstruct onsets and combine (ONLY if dual output)
    final_onset_tensor = None
    if len(all_onset_probs) > 0:
        final_onset_tensor = torch.cat(all_onset_probs, dim=1)[:, :original_frames]
        
    binary_tensor = binarize_output(all_frame_probs, final_onset_tensor, frame_threshold=frame_threshold, onset_threshold=onset_threshold)

    # Smoothing and Post-Processing
    binary_numpy = binary_tensor.cpu().numpy()
    for i in range(binary_numpy.shape[0]):
        binary_numpy[i, :] = medfilt(binary_numpy[i, :], kernel_size=5)
    
    # Convert to MIDI
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    midi_path = output_dir / f"{trained_model.__class__.__name__}_transcribed.midi"
    
    midi_obj = output_to_midi(binary_numpy, sample_rate=sr, hop_length=hop_length)
    midi_obj = snap_midi_timing(midi_obj)
    midi_obj.write(str(midi_path))
    print(f"Saved MIDI to: {midi_path}")
    
    # Guess tempo 
    y = waveform.squeeze().cpu().numpy()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    estimated_bpm = int(tempo[0]) if isinstance(tempo, np.ndarray) else int(tempo)
    
    print("Converting to Sheet Music...")
    try:
        sheet = midi_to_sheet(midi_obj, estimated_bpm)
        sheet_path = output_dir / f"{trained_model.__class__.__name__}_sheet.xml"
        sheet.write("musicxml", fp=str(sheet_path)) #type: ignore
        if show:
            sheet.show() #type: ignore
    except Exception as e:
        print(f"Failed to render sheet music: {e}")


def snap_midi_timing(midi_obj, tolerance_sec=0.04):
    """
    Forces notes that start or end almost simultaneously to align perfectly.
    tolerance_sec=0.04 means notes within 40ms of each other are snapped together.
    """
    for instrument in midi_obj.instruments:
        #Snap Start Times (Onsets)
        notes = sorted(instrument.notes, key=lambda n: n.start)
        if not notes:
            continue
            
        current_start_group = notes[0].start
        for note in notes:
            if abs(note.start - current_start_group) <= tolerance_sec:
                note.start = current_start_group # Fudge the number to match the group
            else:
                current_start_group = note.start # Start a new time group

        # Snap End Times (Offsets) - Prevents tiny overlapping tails
        notes = sorted(instrument.notes, key=lambda n: n.end)
        current_end_group = notes[0].end
        for note in notes:
            if abs(note.end - current_end_group) <= tolerance_sec:
                note.end = current_end_group
            else:
                current_end_group = note.end
                
        instrument.notes = sorted(notes, key=lambda n: n.start)
        
    return midi_obj