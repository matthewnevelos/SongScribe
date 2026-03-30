import random
import torch
import torchaudio
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
        
        self.index_map = []
        #chop audio into segments
        for song_idx, row in enumerate(self.metadata):
            num_segments = int(row.duration // self.segment_seconds)
            
            for seg_idx in range(num_segments):
                start_time = seg_idx * self.segment_seconds
                self.index_map.append((song_idx, start_time))
            
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # FIX: Unpack the correct song index and the start time of the chunk 
        # from the index map to prevent the out-of-bounds IndexError.
        song_idx, start_time_sec = self.index_map[idx]
        row = self.metadata[song_idx]
        

        audio_info = torchaudio.info(row.audio_path)
        orig_sr = audio_info.sample_rate
        total_audio_frames = audio_info.num_frames
        
        # Calculate exact frames to load based on the index map
        start_frame = int(start_time_sec * orig_sr)
        frames_to_load = int(self.segment_seconds * orig_sr)
        
        # Safeguard: If this is the final chunk of the song, don't load past the end
        if start_frame + frames_to_load > total_audio_frames:
            frames_to_load = total_audio_frames - start_frame

        waveform, sr = torchaudio.load(
            row.audio_path, 
            frame_offset=start_frame, 
            num_frames=frames_to_load,
        )
        
        # Ensure waveform is always 2D [Channels, Time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        #if sr != self.sample_rate:
        #    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        #if waveform.shape[0] > 1:
        #    waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Removed after data resampled to speed up training time

        if waveform.shape[1] > self.audio_chunk_frames:
            waveform = waveform[:, :self.audio_chunk_frames]
        elif waveform.shape[1] < self.audio_chunk_frames:
            padding_needed = self.audio_chunk_frames - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        
        cqt_frames = (self.audio_chunk_frames // self.hop_length) + 1

        # Safe path resolution for both tensors
        midi_path = self.tensor_dir / Path(row.midi_filename).with_suffix(".midi.tensor")
        onset_path = Path(str(midi_path).replace('.midi.tensor', '.midi.onset.tensor'))
        
        full_frame_label = torch.load(midi_path, weights_only=True)
        full_onset_label = torch.load(onset_path, weights_only=True)

        # Ensure labels are always 2D [88, Time]
        if full_frame_label.dim() == 1:
                full_frame_label = full_frame_label.unsqueeze(0)
        if full_onset_label.dim() == 1:
                full_onset_label = full_onset_label.unsqueeze(0)

        # Calculate time columns based on the exact start_time_sec
        start_col = int(start_time_sec * self.frames_per_second)
        end_col = start_col + cqt_frames
                
        # Slice both labels
        frame_chunk = full_frame_label[:, start_col:end_col]
        onset_chunk = full_onset_label[:, start_col:end_col]
        
        # STRICT PADDING FOR FRAMES
        if frame_chunk.shape[1] > cqt_frames:
            frame_chunk = frame_chunk[:, :cqt_frames]
        elif frame_chunk.shape[1] < cqt_frames:
            padding_needed = cqt_frames - frame_chunk.shape[1]
            frame_chunk = torch.nn.functional.pad(frame_chunk, (0, padding_needed))
            
        # STRICT PADDING FOR ONSETS
        if onset_chunk.shape[1] > cqt_frames:
            onset_chunk = onset_chunk[:, :cqt_frames]
        elif onset_chunk.shape[1] < cqt_frames:
            padding_needed = cqt_frames - onset_chunk.shape[1]
            onset_chunk = torch.nn.functional.pad(onset_chunk, (0, padding_needed))
            
        return waveform, frame_chunk, onset_chunk