import matplotlib.pyplot as plt
import numpy as np

def plot_debug_cqts(masked_cqt, raw_cqt):
    """
    This is stright from chatgpt, just for a visual on augmented vs raw CQT
    
    Plots the raw and masked CQT spectrograms for visual comparison.
    Expects PyTorch tensors as input.
    """
    # Detach from graph, move to CPU, convert to NumPy, and drop channel dims
    raw_np = raw_cqt.detach().cpu().squeeze().numpy()
    masked_np = masked_cqt.detach().cpu().squeeze().numpy()
    
    # Optional: Convert to log scale (dB) for better visualization if not already in log scale
    raw_np = 10 * np.log10(raw_np + 1e-8)
    masked_np = 10 * np.log10(masked_np + 1e-8)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

    # Plot Raw CQT
    img1 = axes[0].imshow(raw_np, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Raw CQT")
    axes[0].set_ylabel("Frequency Bins (0-87)")
    fig.colorbar(img1, ax=axes[0], label="Magnitude")

    # Plot Masked CQT
    img2 = axes[1].imshow(masked_np, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Augmented & Masked CQT")
    axes[1].set_ylabel("Frequency Bins (0-87)")
    axes[1].set_xlabel("Time Frames")
    fig.colorbar(img2, ax=axes[1], label="Magnitude")

    plt.tight_layout()
    plt.show()