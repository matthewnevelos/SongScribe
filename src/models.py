import torch
import torch.nn as nn

class ConvStack(nn.Module):
    
    # Layers in the model

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
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
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class PianoTranscriptionCRNN(nn.Module):

    # This is the onsets and frames architecture
    # Now measures when notes start and also when a note is sustaining
    # Should make the model better at seeing multiple notes of the same pitch that are played one after the other

    def __init__(self, freq_bins=88):
        super().__init__()
        
        # 1. Shared Feature Extractor
        self.acoustic_model = ConvStack()
        
        final_freq_bins = freq_bins // 4
        cnn_features = 128 * final_freq_bins # 2816 features
        
        # 2. Onset Head
        self.onset_rnn = nn.GRU(cnn_features, 256, bidirectional=True, batch_first=True)
        self.onset_linear = nn.Linear(512, 88)
        
        # 3. Frame (Sustain) Head
        # Notice the input size: It takes the CNN features PLUS the 88 Onset predictions
        self.frame_rnn = nn.GRU(cnn_features + 88, 256, bidirectional=True, batch_first=True)
        self.frame_linear = nn.Linear(512, 88)

    def forward(self, x):
        # --- Shared CNN ---
        # x shape: (Batch, 1, Freq, Time)
        x = self.acoustic_model(x) 
        B, C, F, T = x.shape
        cnn_out = x.view(B, C * F, T).transpose(1, 2) # -> (Batch, Time, Features)
        
        # --- Onset Head ---
        onset_rnn_out, _ = self.onset_rnn(cnn_out)
        onset_logits = self.onset_linear(onset_rnn_out) # -> (Batch, Time, 88)
        
        # To feed onsets into the frame head during training, we apply sigmoid 
        # to get them into a [0,1] probability range. Stop gradients so the frame 
        # head doesn't mess up the onset head's learning.
        onset_probs = torch.sigmoid(onset_logits).detach() 
        
        # --- Frame Head ---
        # Concatenate CNN features with Onset probabilities
        frame_input = torch.cat([cnn_out, onset_probs], dim=-1) 
        
        frame_rnn_out, _ = self.frame_rnn(frame_input)
        frame_logits = self.frame_linear(frame_rnn_out) # -> (Batch, Time, 88)
        
        # Transpose back to (Batch, 88, Time) for the loss function
        return onset_logits.transpose(1, 2), frame_logits.transpose(1, 2)