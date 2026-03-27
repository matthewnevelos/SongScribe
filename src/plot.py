import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from pathlib import Path

def save_debug_samples(raw_waveform, augmented_waveform, raw_cqt, augmented_cqt, sample_rate, save_dir="debug_output", prefix="batch0_item0"):
    """
    Saves the raw and augmented audio files and plots their CQT spectrograms
    """
    #setupthe output directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    raw_audio_out = raw_waveform.detach().cpu()
    aug_audio_out = augmented_waveform.detach().cpu()
    
    if raw_audio_out.ndim == 1:
        raw_audio_out = raw_audio_out.unsqueeze(0)
    if aug_audio_out.ndim == 1:
        aug_audio_out = aug_audio_out.unsqueeze(0)

    torchaudio.save(save_path / f"{prefix}_raw.wav", raw_audio_out, sample_rate)
    torchaudio.save(save_path / f"{prefix}_augmented.wav", aug_audio_out, sample_rate)

    raw_np = raw_cqt.detach().cpu().squeeze().numpy()
    masked_np = augmented_cqt.detach().cpu().squeeze().numpy()
    
    # convert to dB
    raw_np = 10 * np.log10(raw_np + 1e-4)
    masked_np = 10 * np.log10(masked_np + 1e-4)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

    # plot Raw CQT
    img1 = axes[0].imshow(raw_np, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Raw CQT")
    axes[0].set_ylabel("Frequency Bins")
    fig.colorbar(img1, ax=axes[0], label="Magnitude (dB)")

    # plot Augmented CQT
    img2 = axes[1].imshow(masked_np, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Augmented CQT")
    axes[1].set_ylabel("Frequency Bins")
    axes[1].set_xlabel("Time Frames")
    fig.colorbar(img2, ax=axes[1], label="Magnitude (dB)")

    plt.tight_layout()
    
    plot_file = save_path / f"{prefix}_cqt_comparison.png"
    plt.savefig(plot_file, dpi=300)
    plt.close(fig) # Free memory
    
    print(f"Saved audio and plots to {save_path.absolute()}")