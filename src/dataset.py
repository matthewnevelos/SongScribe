import random
import torch
import torchaudio
from src.preprocess import load_metadata, MaestroPreprocessor
from torch.utils.data import Dataset
from pathlib import Path

class MaestroDataset(Dataset):
    def __init__(self, maestro_dir, preprocessor:MaestroPreprocessor, segment_seconds=5.0):
        self.preprocessor = preprocessor
        self.segment_seconds = segment_seconds
        
        self.sample_rate = self.preprocessor.target_sr
        self.frames_per_second = self.preprocessor.frames_per_seconds
        
        self.tensor_dir = Path(maestro_dir).parent / f"_{preprocessor.target_sr}"
        
        self.audio_chunk_frames = int(self.segment_seconds * self.sample_rate)
        self.label_chunk_frames = int(self.segment_seconds * self.frames_per_second)

        # Read the CSV into a list of dictionaries
        self.metadata = load_metadata(maestro_dir)
        
        # preprocess midi if not exist
        out_dir = Path(maestro_dir).parent / f"midi_{self.preprocessor.target_sr}"
        if not out_dir.exists():
            self.preprocessor.precompute_midi_labels(out_dir, self.metadata)
            

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]

        # randomly slice song 
        audio_info = torchaudio.info(row.audio_path)
        total_audio_frames = audio_info.num_frames
        
        if total_audio_frames > self.audio_chunk_frames:
            start_frame = random.randint(0, total_audio_frames - self.audio_chunk_frames)
        else:
            start_frame = 0

        #load just the `segment_seconds` second chunk
        waveform, sr = torchaudio.load(
            row.audio_path, 
            frame_offset=start_frame, 
            num_frames=self.audio_chunk_frames
        )
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # apply self.preprocessor
        spectrogram = self.preprocessor(waveform, orig_sr=self.sample_rate, augment=True) 
        # Will we ever want augment != True? like during testing? will this dataset be used?
        
        #load and slice MIDI piano roll
        start_time_sec = start_frame / audio_info.sample_rate
        
        cqt_frames = spectrogram.shape[-1]
        
        midi_path = self.tensor_dir / Path(row.midi_filename).with_suffix(".midi.tensor")
        full_label = torch.load(midi_path)
        
        start_time_sec = start_frame / audio_info.sample_rate
        start_col = int(start_time_sec * self.frames_per_second)
        end_col = start_col + cqt_frames
        
        label_chunk = full_label[:, start_col:end_col]
        
        if label_chunk.shape[1] < cqt_frames:
            padding_size = cqt_frames - label_chunk.shape[1]
            padding = torch.zeros((88, padding_size), dtype=torch.float32)
            label_chunk = torch.cat((label_chunk, padding), dim=1)

        return spectrogram, label_chunk