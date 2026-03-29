from preprocess import CQTPreprocessor
import librosa
import numpy as np
import pretty_midi
import music21
import io
import functools


@functools.lru_cache(maxsize=1)
def get_cqt_preprocessor(device='cpu', sample_rate=22050, hop_length=256, f_min=27.5, n_bins=84):
    # cache preprocessor so I dont rebuild ever time audio_to_CQT is called
    preprocessor = CQTPreprocessor(
        target_sr=sample_rate, 
        hop_length=hop_length, 
        f_min=f_min, 
        n_bins=n_bins
    )
    return preprocessor.to(device)

def audio_to_CQT(audio, sample_rate=22050, hop_length=256, f_min=27.5, n_bins=84):
    current_device = str(audio.device) if hasattr(audio, 'device') else 'cpu'
    preprocessor = get_cqt_preprocessor(
        device=current_device,
        sample_rate=sample_rate,
        hop_length=hop_length,
        f_min=f_min,
        n_bins=n_bins
    )
    return preprocessor(audio, sample_rate)


def cqt_to_audio(cqt, sample_rate=22050, hop_length=256, f_min=27.5, bins_per_octave=12):
    #shape is (Batch, Bins, Time)
    cqt_numpy = cqt.squeeze(0).cpu().detach().numpy()

    # inverse CQT
    audio = librosa.griffinlim_cqt(
        cqt_numpy,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=f_min,
        bins_per_octave=bins_per_octave
    )
    
    return audio



def output_to_midi(raw_output, threshold=0.5, sample_rate = 22050, hop_length = 256):
    frames_per_second = sample_rate / hop_length
    midi = pretty_midi.PrettyMIDI()
    right_hand = pretty_midi.Instrument(program=0, name="Treble")
    left_hand = pretty_midi.Instrument(program=0, name="Bass")
    
    binary_matrix = (raw_output > threshold).astype(int)
    
    for pitch_idx in range(88):
        midi_pitch = pitch_idx + 21 
        
        active_frames = binary_matrix[pitch_idx, :]
        padded_frames = np.pad(active_frames, (1, 1), mode='constant')
        
        diff = np.diff(padded_frames)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start_frame, end_frame in zip(starts, ends):
            start_time = start_frame / frames_per_second
            end_time = end_frame / frames_per_second
            
            note = pretty_midi.Note(
                velocity=100, 
                pitch=midi_pitch, 
                start=start_time, 
                end=end_time
            )
            
        # Split hands at Middle C (Pitch 60)
            if midi_pitch >= 60:
                right_hand.notes.append(note)
            else:
                left_hand.notes.append(note)
                
    midi.instruments.append(right_hand)
    midi.instruments.append(left_hand)  
    return midi


def midi_to_sheet(midi, bpm):
    buffer = io.BytesIO()
    midi.write(buffer) # write to buffer instead of saving file
    buffer.seek(0)
    
    score = music21.converter.parse(buffer.read(), format="midi")
    
    mm = music21.tempo.MetronomeMark(number=bpm)
    score.insert(0, mm)
    
    ts = music21.meter.TimeSignature('4/4') # western music is usually 4/4, hard code for now. #type: ignore
    score.insert(0, ts)
    
    score.quantize([1, 2, 4, 8, 16], inPlace=True)

    score.show() 