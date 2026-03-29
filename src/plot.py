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
    
def plot_CQT(CQT, show=False, file_path = None, title=None):
    arr = CQT.detach().cpu().squeeze().numpy()
    
    # convert to dB
    arr = 10 * np.log10(arr + 1e-4)

    fig, ax = plt.subplots(figsize=(10, 4))

    img1 = ax.imshow(arr, aspect='auto', origin='lower', cmap='magma')
    
    # Labeling
    if title:
        ax.set_title(title)
    ax.set_ylabel("Frequency Bins")
    ax.set_xlabel("Time Frames")
    fig.colorbar(img1, ax=ax, label="Magnitude (dB)")

    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=300)

    if show:
        plt.show()
        
    plt.close(fig)
        
def plot_sheet(sheet, show=False, file_path = None):
    if show:
        sheet.show()

    if file_path:
        sheet.write("musicxml", fp=str(file_path))


import matplotlib.pyplot as plt

def plot_midi(midi, show=False, file_path=None, title=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # colours for using multiple instruments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    max_time = 0
    min_pitch = 108 # highest piano key
    max_pitch = 21  # lowest piano key
    
    for i, instrument in enumerate(midi.instruments):
        color = colors[i % len(colors)]
        label_added = False
        
        for note in instrument.notes:
            ax.plot(
                [note.start, note.end], 
                [note.pitch, note.pitch], 
                color=color, 
                linewidth=3, 
                solid_capstyle='butt', #Makes the ends of the lines flat
                label=instrument.name if not label_added else None
            )
            label_added = True
            
            # Track the bounds of the song to size the graph perfectly
            max_time = max(max_time, note.end)
            min_pitch = min(min_pitch, note.pitch)
            max_pitch = max(max_pitch, note.pitch)

    if max_time > 0:
        ax.set_xlim(0, max_time + 0.5)
        ax.set_ylim(max(21, min_pitch - 3), min(108, max_pitch + 3))
    else:
        ax.set_xlim(0, 5)
        ax.set_ylim(21, 108)

    # Formatting and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Pitch")
    
    ax.grid(True, axis='both', linestyle='--', alpha=0.5)
    
    # Add a legend if we actually plotted named instruments
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='upper right')

    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=300)
        print(f"Saved MIDI plot to: {file_path}")

    if show:
        plt.show()

    plt.close(fig)