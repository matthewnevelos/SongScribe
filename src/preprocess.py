from pathlib import Path
import csv
from dataclasses import dataclass
import torch
import torchaudio
import torchvision
import torch.nn.functional as F
from nnAudio.features.cqt import CQT1992v2
import pretty_midi
import numpy as np



@dataclass
class MaestroMetadata:
    """
    dataclass to manage each songs info
    """
    composer: str
    title: str
    split: str
    year: int
    midi_filename: str
    midi_path: Path
    audio_filename: str
    audio_path: Path
    duration: float



def load_metadata(maestro_path: str) -> list[MaestroMetadata]:
    dataset = []
    
    csv_path = Path(maestro_path) / "maestro-v3.0.0.csv"
    
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            item = MaestroMetadata(
                composer = row["canonical_composer"],
                title = row["canonical_title"],
                split = row["split"],
                year = int(row["year"]),
                
                midi_filename = row["midi_filename"],
                midi_path = Path(maestro_path) / row["midi_filename"], # resolved path
                audio_filename = row["audio_filename"],
                audio_path = Path(maestro_path) / row["audio_filename"],
                
                duration = float(row["duration"])
            )
            dataset.append(item)
    
    return dataset



class WaveformAugmenter(torch.nn.Module):
    def __init__(self, min_gain=0.5, max_gain=1.5, max_noise_factor=0.01):
        """
        control input gain and background sound.
        
        Parameters
        ----------
        min_gain : float, optional
            min volume multiplier, by default 0.5
        max_gain : float, optional
            max volume multiplier, by default 1.5
        max_noise_factor : float, optional
            background noise amplitude, by default 0.01
        """

        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.max_noise_factor = max_noise_factor

    def forward(self, waveform, augment=True):   
        device = waveform.device
        
        if not augment:
            return waveform

        # scale gain
        gain_multiplier = self.min_gain + torch.rand(1, device=device) * (self.max_gain - self.min_gain)
        augmented_waveform = waveform * gain_multiplier
        
        # noise
        noise_factor = torch.rand(1, device=device) * self.max_noise_factor #TODO play with different random distributions
        
        # add to waveform
        white_noise = torch.randn_like(augmented_waveform) * noise_factor
        augmented_waveform = augmented_waveform + white_noise
        
        # clamp between -1 and 1
        augmented_waveform = torch.clamp(augmented_waveform, min=-1.0, max=1.0)
        
        return augmented_waveform
    
    
    
class CQTPreprocessor(torch.nn.Module):
    def __init__(self, target_sr=22050, hop_length=256, f_min=27.5, n_bins=84):
        """
        GPU accelerated CQT preprocessing

        Parameters
        ----------
        target_sr : int, optional
            Target sampling rate (captures frequencies up to target_sr/2), by default 22050
        hop_length : int, optional
            Number of time samples in each frame, by default 256
        f_min : float, optional
            minimum frequency
        n_bins : int, optional
            number of frequency bins, by default 84
        """

        super().__init__()
        self.target_sr = target_sr
        
        self.cqt = CQT1992v2(sr=target_sr, 
                             hop_length=hop_length, 
                             fmin=f_min, 
                             n_bins=n_bins, 
                             bins_per_octave=12) #TODO try 2010v2 and check speed/performance
        
        # Cache resamplers
        self.resamplers = {}

    def forward(self, waveform, orig_sr):
        """
        takes a raw audio waveform and returns the CQT spectrogram.
        """
        device = waveform.device
        
        # convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # downsample
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.target_sr).to(device)
        waveform = self.resamplers[orig_sr](waveform)
            
        cqt_spectrogram = self.cqt(waveform)
        
        return cqt_spectrogram
    
class _SpectAxisAugmenter(torch.nn.Module):
    def __init__(self, axis, sparsity = 0.1):
        """
        _summary_

        Parameters
        ----------
        axis : int
            -2 for frquency, -1 for time
        sparsity : float
            ratio of lines zeroed out [0.0, 1)
        """
        super().__init__()
        self.axis = axis
        self.sparsity = sparsity
    
    def forward(self, spectrogram, augment = True):
        if not augment or self.sparsity <= 0.0:
            return spectrogram

        masked = spectrogram.clone()
        axis_size = masked.shape[self.axis]
        
        probabilities = torch.rand(axis_size, device=masked.device)
        lines_to_drop = probabilities < self.sparsity

            
        if self.axis == -1: # time axis
            masked[..., :, lines_to_drop] = 0.0
        elif self.axis == -2: # frequency axis
            masked[..., lines_to_drop, :] = 0.0
            
        return masked
        
    
class SpectFreqAugmenter(_SpectAxisAugmenter):
    def __init__(self, sparsity):
        super().__init__(sparsity=sparsity, axis=-2)
    
class SpectTimeAugmenter(_SpectAxisAugmenter):
    def __init__(self, sparsity):
        super().__init__(sparsity=sparsity, axis=-1)


class SpectChunkAugmenter(torch.nn.Module):
    def __init__(self, sparsity=0.1, freq_size = (2,8), time_size = (5,20)):        
        """
        DropBlock algorithm designed with rectangular blocks. 
        largely the same as https://docs.pytorch.org/vision/main/_modules/torchvision/ops/drop_block.html#DropBlock2d

        - create tensor of 0's
        - randomly place `gamma` number of seeds as 1's in the tensor.
        - use max pooling to "grow" them to chunk_freq x chunk_time sized blocks of 1's
        
        Parameters
        ----------
        sparsity : float
            approximate percentage of the spectrogram to mask [0, 1)
        freq_size : tuple[int, int]
            The min/max height of the chunks
        time_size : tuple[int, int]
            The min/max width of the chunks
        """
        super().__init__()
        self.sparsity = sparsity
        self.min_freq, self.max_freq = freq_size
        self.min_time, self.max_time = time_size

    def forward(self, spectrogram, augment=True):
        if not augment or self.sparsity <= 0.0:
            return spectrogram

        masked = spectrogram.clone()
        orig_shape = masked.shape
        
        while masked.ndim < 4:
            masked = masked.unsqueeze(0)

        # randomly select chunk size
        chunk_freq = int(torch.randint(self.min_freq, self.max_freq + 1, (1,)).item())
        chunk_time = int(torch.randint(self.min_time, self.max_time + 1, (1,)).item())

        chunk_area = chunk_freq * chunk_time
        gamma = self.sparsity / chunk_area

        #place seeds
        seeds = (torch.rand_like(masked) < gamma).float()

        pad_left, pad_right = 0, chunk_time - 1
        pad_top, pad_bottom = 0, chunk_freq - 1
        
        padded_seeds = F.pad(seeds, (pad_left, pad_right, pad_top, pad_bottom))
        
        # grow seeds with pooling
        chunk_mask = F.max_pool2d(
            padded_seeds, 
            kernel_size=(chunk_freq, chunk_time), 
            stride=1
        )

        masked = masked * (1.0 - chunk_mask)
        return masked.view(orig_shape)

class SpectAugmenter(torch.nn.Module):
    def __init__(self, freq_sparsity, time_sparsity, chunk_sparsity, freq_chunk_size, time_chunk_size):
        """
        mask lines of constant freqeucny  / time based on random sparsity.
        """
        super().__init__()
                
        self.freq_mask = SpectFreqAugmenter(sparsity = freq_sparsity)
        self.time_mask = SpectTimeAugmenter(sparsity = time_sparsity)
        self.chunk_mask = SpectChunkAugmenter(sparsity = chunk_sparsity, freq_size=freq_chunk_size, time_size=time_chunk_size)
        
    def forward(self, spectrogram, augment=True):
        if not augment:
            return spectrogram
        masked = self.time_mask(spectrogram).to(spectrogram.device)
        masked = self.freq_mask(masked).to(spectrogram.device)
        masked = self.chunk_mask(masked).to(spectrogram.device)
        return masked
    
class MaestroPreprocessor(torch.nn.Module):
    def __init__(
        self,
        min_gain = 0.5, #waveform aug
        max_gain = 1.5,
        max_noise_factor = 0.01,
        waveform_aug_p = 1,
        
        target_sr = 22050, #CQT
        hop_length = 256,
        f_min = 27.5,
        n_bins = 88,
        
        freq_sparsity = 0.1, #CQT augment
        time_sparsity = 0.2, 
        chunk_sparsity = 0.2, 
        freq_chunk_size = (1, 5), 
        time_chunk_size = (1, 20),
        spectrogram_aug_p = 1,
        ):
        """
        Maestro Preprocessing

        Parameters
        ----------
        min_gain : float, optional
            min volume multiplier, by default 0.5
        max_gain : float, optional
            max volume multiplier, by default 1.5
        max_noise_factor : float, optional
            background noise amplitude, by default 0.01
        waveform_aug_p : float, optional
            probability of augmenting waveform [0,1], by default 0.5
            
        target_sr : int, optional
            Target sampling rate (captures frequencies up to target_sr/2), by default 22050
        hop_length : int, optional
            Number of time samples in each frame, by default 256
        f_min : float, optional
            minimum frequency
        n_bins : int, optional
            number of frequency bins, by default 84
            
        freq_sparsity : float, optional
            sparsity of frequency rows, by default 0.1
        time_sparsity : float, optional
            sparsity of time columns, by default 0.2
        chunk_sparsity : float, optional
            average area of chunk mask, by default 0.2
        freq_chunk_size : tuple[float, float], optional
            bounds on chunk height
        time_chunk_size : tuple[float, float], optional
            bounds on chunk width
        spectrogram_aug_p : float, optional
            probability of augmenting spectrogram [0,1], by default 0.5
        """
        super().__init__()
        
        self.target_sr = target_sr
        self.hop_length = hop_length
        self.waveform_aug_p = waveform_aug_p
        self.spectrogram_aug_p = spectrogram_aug_p

        self.wave_augmenter = WaveformAugmenter(
            min_gain=min_gain, 
            max_gain=max_gain, 
            max_noise_factor=max_noise_factor
        )
        self.cqt = CQTPreprocessor(
            target_sr=target_sr, 
            hop_length=hop_length, 
            f_min=f_min, 
            n_bins=n_bins
        )
        self.spec_aug = SpectAugmenter(
            freq_sparsity=freq_sparsity, 
            time_sparsity=time_sparsity, 
            chunk_sparsity=chunk_sparsity, 
            freq_chunk_size=freq_chunk_size, 
            time_chunk_size=time_chunk_size
        )
        
        
    def process_audio(self, waveform, orig_sr, augment=True, debug = False):
        """
        return masked_cqt, unless debug is true, then return masked_cqt and raw_cqt.
        """
        random = torch.rand(2)
        augmented = waveform #augmented tensor
        
        # augment waveform
        if augment and random[0] < self.waveform_aug_p:
            augmented = self.wave_augmenter(augmented, augment)
            
        # do CQT    
        augmented = self.cqt(augmented, orig_sr)
        
        # augment spectrogram
        if augment and random[1] < self.spectrogram_aug_p: #TODO split augmentation probability into time, freq, chunk
            augmented = self.spec_aug(augmented, augment)
        
        if debug:
            raw_cqt = self.cqt(waveform, orig_sr)
            return augmented, raw_cqt
        return augmented
    
    
    def process_midi(self, midi_path, audio_frames):
        """
        converts MIDI file to binary 2D PyTorch tensor aligned to the audio CQT.
        """
        fs = self.target_sr / self.hop_length # sampling frequency same as CQT
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        #  defaults to 128 notes
        piano_roll = pm.get_piano_roll(fs=fs) # type: ignore
        piano_roll_88 = piano_roll[21:109, :] # slice 88 pianos keys (A0=MIDI note 21, C8 = 108)
        
        #binarize notes. 1=on, 0=off
        binary_roll = (piano_roll_88 > 0).astype(np.float32) #TODO try and keep note velocity. might be too complicated
        label_tensor = torch.from_numpy(binary_roll)
        
        # truncate/pad midi to match audio if need be.
        midi_frames = label_tensor.shape[1]
        
        if midi_frames > audio_frames:
            label_tensor = label_tensor[:, :audio_frames]
        elif midi_frames < audio_frames:
            padding = int (audio_frames - midi_frames)
            label_tensor = torch.nn.functional.pad(label_tensor, (0, padding))
            
        label_tensor = label_tensor.unsqueeze(0) # match 1 channel dimension [1, 88, audio_frames]
        
        return label_tensor
        
        
    def forward(self, waveform, orig_sr, midi_path=None, augment=True, debug=False):
        x_cqt = self.process_audio(waveform=waveform, orig_sr=orig_sr, augment=augment, debug=debug)
        
        if midi_path is None:
            return x_cqt
        
        if debug:
            audio_frames = x_cqt[0].shape[-1]
        else:
            audio_frames = x_cqt.shape[-1] #type: ignore
        
        y_cqt = self.process_midi(midi_path, audio_frames)
        
        if debug:
            return x_cqt, y_cqt
        
        return x_cqt, y_cqt
