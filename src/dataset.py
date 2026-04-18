import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class MaestroDataset(Dataset):
    def __init__(self, maestro_dir, target_sr, hop_length, metadata, segment_seconds=5.0, return_onsets=False):
        """Dataset class for MAESTRO dataset"""
        self.segment_seconds = segment_seconds
        self.sample_rate = target_sr
        self.hop_length = hop_length
        self.metadata = metadata
        self.return_onsets = return_onsets
        
        self.frames_per_second = self.sample_rate / self.hop_length
        
        self.tensor_dir = Path(maestro_dir).parent / f"maestro_{self.sample_rate}"
        
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
        return int(len(self.index_map))

    def __getitem__(self, idx):
        song_idx, start_time = self.index_map[idx]
        song = self.metadata[song_idx]
        
        # laod precomputed audio
        precomputed_audio_path = self.tensor_dir / Path(song.audio_filename).with_suffix('.wav')
        
        audio_info = torchaudio.info(precomputed_audio_path)
        orig_sr = audio_info.sample_rate
        total_audio_frames = audio_info.num_frames
        
        start_frame = int(orig_sr * start_time)
        frames_to_load = int(self.segment_seconds * orig_sr)
        
        if start_frame + frames_to_load > total_audio_frames:
            frames_to_load = total_audio_frames - start_frame
        
        waveform, _ = torchaudio.load(precomputed_audio_path, frame_offset=start_frame, num_frames=frames_to_load)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        if waveform.shape[1] > self.audio_chunk_frames:
            waveform = waveform[:, :self.audio_chunk_frames]
        elif waveform.shape[1] < self.audio_chunk_frames:
            padding_needed = self.audio_chunk_frames - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        
        cqt_frames = (self.audio_chunk_frames // self.hop_length) + 1

        midi_path = self.tensor_dir / Path(song.midi_filename).with_suffix(".midi.tensor")
        combined_label = torch.load(midi_path, weights_only=True)
        
        if combined_label.dim() == 1:
            combined_label = combined_label.unsqueeze(0)

        # Calculate columns
        start_time_sec = start_frame / orig_sr
        start_col = int(start_time_sec * self.frames_per_second)
        end_col = start_col + cqt_frames
                
        combined_chunk = combined_label[:, start_col:end_col]
        
        if combined_chunk.shape[1] > cqt_frames:
            combined_chunk = combined_chunk[:, :cqt_frames]
        elif combined_chunk.shape[1] < cqt_frames:
            padding_needed = cqt_frames - combined_chunk.shape[1]
            combined_chunk = torch.nn.functional.pad(combined_chunk, (0, padding_needed))
            
        # Extract Binary Masks
        frame_chunk = (combined_chunk > 0).float()
            
        if not self.return_onsets:
            return waveform, frame_chunk
        
        onset_chunk = (combined_chunk == 1).float()

        return waveform, frame_chunk, onset_chunk