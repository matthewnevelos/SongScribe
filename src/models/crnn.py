import torch
import torch.nn as nn

class PianoTranscriptionCRNN_v1(nn.Module):
    def __init__(self, freq_bins=88):
        super().__init__()
        
        # 1. Deeper CNN with Dropout
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            nn.Dropout(0.2), # Add Dropout
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.2),
            
            # 3rd Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # No pooling here to preserve enough frequency resolution
            nn.Dropout(0.2)
        )
        
        # Calculate remaining frequency bins after two MaxPool2d(2,1)
        final_freq_bins = freq_bins // 4
        cnn_output_features = 128 * final_freq_bins # 128 channels * 22 bins = 2816
        
        # 2. Bidirectional GRU for Temporal Smoothing
        # hidden_size=256 per direction means output will be 512 features
        self.rnn = nn.GRU(
            input_size=cnn_output_features, 
            hidden_size=256, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Classifier 
        # Maps the 512 RNN features to the 88 piano keys
        self.classifier = nn.Linear(512, 88)

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)
        x = self.conv_blocks(x) # -> (Batch, 128, 22, Time)
        
        # Flatten channels and frequency
        B, C, F, T = x.shape
        x = x.view(B, C * F, T) # -> (Batch, 2816, Time)
        
        # PyTorch RNNs expect shape: (Batch, Time, Features)
        x = x.transpose(1, 2) # -> (Batch, Time, 2816)
        
        # Pass through RNN
        x, _ = self.rnn(x) # -> (Batch, Time, 512)
        
        # Project to 88 keys
        x = self.classifier(x) # -> (Batch, Time, 88)
        
        # Transpose back to match your Loss Function and labels: (Batch, 88, Time)
        x = x.transpose(1, 2) 
        
        return x
    
    
class PianoTranscriptionCRNN_v2(nn.Module):
    def __init__(self, freq_bins=88):
        super().__init__()
        
        # 1. Deeper CNN with Dropout
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(0.2),
            
            # 3rd Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Calculate remaining frequency bins after two MaxPool2d(2,1)
        final_freq_bins = freq_bins // 4
        cnn_output_features = 128 * final_freq_bins # 128 channels * 22 bins = 2816
        
        # 2. FEATURE BOTTLENECK (New Addition)
        # Compresses the massive CNN output before the RNN to save memory and prevent overfitting
        self.bottleneck = nn.Sequential(
            nn.Linear(cnn_output_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Upgraded Bidirectional GRU
        # Input is now 256 (from bottleneck). Increased to 2 layers with internal dropout.
        self.rnn = nn.GRU(
            input_size=256, 
            hidden_size=256, 
            num_layers=2,           # Increased depth for better sustain/timing tracking
            batch_first=True, 
            bidirectional=True,
            dropout=0.2             # Applies dropout between GRU layers
        )
        
        # 4. Classifier 
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),        # Pre-classifier dropout
            nn.Linear(512, 88)      # 512 because bidirectional (256 * 2)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)
        x = self.conv_blocks(x) # -> (Batch, 128, 22, Time)
        
        # Flatten channels and frequency
        B, C, F, T = x.shape
        x = x.view(B, C * F, T) # -> (Batch, 2816, Time)
        
        # PyTorch Linear/RNNs expect Time in the middle: (Batch, Time, Features)
        x = x.transpose(1, 2)   # -> (Batch, Time, 2816)
        
        # Pass through bottleneck
        x = self.bottleneck(x)  # -> (Batch, Time, 256)
        
        # Pass through RNN
        x, _ = self.rnn(x)      # -> (Batch, Time, 512)
        
        # Project to 88 keys
        x = self.classifier(x)  # -> (Batch, Time, 88)
        
        # Transpose back to match Loss Function and labels: (Batch, 88, Time)
        x = x.transpose(1, 2) 
        
        return x