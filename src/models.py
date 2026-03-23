import torch
import torch.nn as nn

class PianoTranscriptionCNN(nn.Module):
    def __init__(self, freq_bins=84): # Default CQT bins is often 84
        super().__init__()
        
        # Input shape expected: (Batch, Channels=1, Freq, Time)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Pool frequency, preserve time
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        
        # After two max-pools of factor 2, the frequency dimension is divided by 4
        final_freq_bins = freq_bins // 4
        
        # A 1D Convolution acts as a classifier across the time steps
        self.classifier = nn.Conv1d(
            in_channels=64 * final_freq_bins, 
            out_channels=88, # The 88 piano keys
            kernel_size=1
        )

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)
        x = self.conv_blocks(x) # -> (Batch, 64, Freq//4, Time)
        
        # Flatten the channel and frequency dimensions together
        B, C, F, T = x.shape
        x = x.view(B, C * F, T) # -> (Batch, 64 * Freq//4, Time)
        
        # Project to 88 keys per time step
        x = self.classifier(x) # -> (Batch, 88, Time)
        return x