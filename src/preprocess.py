from pathlib import Path
import csv
from dataclasses import dataclass
import torch
import torchaudio
from nnAudio.features.cqt import CQT1992v2


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
    def __init__(self, p=0.5, min_gain=0.5, max_gain=1.5, max_noise_factor=0.01):
        """
        Applies random time-domain augmentations to raw audio tensors.
        
        Args:
            p: Probability that the augmentations will be applied (0.0 to 1.0).
            min_gain: Minimum volume multiplier.
            max_gain: Maximum volume multiplier.
            max_noise_factor: Maximum amplitude of the injected white noise.
        """
        super().__init__()
        self.p = p
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.max_noise_factor = max_noise_factor

    def forward(self, waveform):
        # Only apply augmentations randomly based on probability 'p'
        # (You only want to augment the Training dataset, never the Test dataset)
        if torch.rand(1).item() > self.p:
            return waveform
            
        device = waveform.device

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
    

class SpectAugmenter(torch.nn.Module):
    def __init__(self, freq_mask_param = 2, time_mask_param = 10, mask_p = 0.5):
        """
        Mask some of the scpectrogram

        Parameters
        ----------
        freq_mask_param : int
            max consectutive masked frequencies
        time_mask_param : int
            max length of mask
        mask_p : float
            proportion of time to be masked
        """
        super().__init__()
                
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param, p=mask_p)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        
    def forward(self, spectrogram):
        # device = spectrogram.device #do I need to(device)? #TODO
        masked = self.time_mask(spectrogram)
        masked = self.freq_mask(masked)
        return masked