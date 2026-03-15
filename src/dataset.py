import random
import torch
import torchaudio
from src.preprocess import load_metadata, MaestroPreprocessor
from torch.utils.data import Dataset
from pathlib import Path

class MaestroDataset(Dataset):
    def __init__(self, maestro_dir, target_sr, hop_length, metadata, segment_seconds=5.0):
        self.segment_seconds = segment_seconds
        self.sample_rate = target_sr
        self.hop_length = hop_length
        self.metadata = metadata
        
        self.frames_per_second = self.sample_rate / self.hop_length
        
        self.tensor_dir = Path(maestro_dir).parent / f"midi_{self.sample_rate}"
        
        self.audio_chunk_frames = int(self.segment_seconds * self.sample_rate)
        self.label_chunk_frames = int(self.segment_seconds * self.frames_per_second)
            

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]

        # randomly slice song 
        audio_info = torchaudio.info(row.audio_path)
        orig_sr = audio_info.sample_rate
        total_audio_frames = audio_info.num_frames
        
        frames_to_load = int(self.segment_seconds * orig_sr)
                
        if total_audio_frames > frames_to_load:
            start_frame = random.randint(0, total_audio_frames - frames_to_load)
        else:
            start_frame = 0

        #load just the `segment_seconds` chunk
        waveform, sr = torchaudio.load(
            row.audio_path, 
            frame_offset=start_frame, 
            num_frames=frames_to_load,
        )
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] > self.audio_chunk_frames:
            waveform = waveform[:, :self.audio_chunk_frames]
        elif waveform.shape[1] < self.audio_chunk_frames:
            padding_needed = self.audio_chunk_frames - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        
        cqt_frames = (self.audio_chunk_frames // self.hop_length) +1

        midi_path = self.tensor_dir / Path(row.midi_filename).with_suffix(".midi.tensor")
        full_label = torch.load(midi_path, weights_only=True)

        #load and slice MIDI piano roll
        start_time_sec = start_frame / orig_sr
        start_col = int(start_time_sec * self.frames_per_second)
        end_col = start_col + cqt_frames
                
        label_chunk = full_label[:, start_col:end_col]
        
        if label_chunk.shape[1] < cqt_frames:
            padding_size = cqt_frames - label_chunk.shape[1]
            padding = torch.zeros((88, padding_size), dtype=torch.float32)
            label_chunk = torch.cat((label_chunk, padding), dim=1)

        return waveform.clone(), label_chunk.clone()