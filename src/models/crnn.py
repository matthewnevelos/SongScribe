import torch
import torch.nn as nn

__all__ = ["PianoTranscriptionCRNN_v1", "PianoTranscriptionCRNN_v2", "PianoTranscriptionCRNN_v3"]

class PianoTranscriptionCRNN_v1(nn.Module):
    def __init__(self, freq_bins=88):
        super().__init__()
        
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
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        final_freq_bins = freq_bins // 4
        cnn_output_features = 128 * final_freq_bins # 128 channels * 22 bins = 2816
        
        self.rnn = nn.GRU(
            input_size=cnn_output_features, 
            hidden_size=256, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.classifier = nn.Linear(512, 88)

    def forward(self, x):
        x = self.conv_blocks(x) # -> (Batch, 128, 22, Time)
        
        B, C, F, T = x.shape
        x = x.view(B, C * F, T) # -> (Batch, 2816, Time)
        
        x = x.transpose(1, 2) # -> (Batch, Time, 2816)
        
        x, _ = self.rnn(x) # -> (Batch, Time, 512)
        
        x = self.classifier(x) # -> (Batch, Time, 88)
        
        x = x.transpose(1, 2) 
        
        return x
    
    
class PianoTranscriptionCRNN_v2(nn.Module):
    def __init__(self, freq_bins=88):
        super().__init__()
        
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
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        final_freq_bins = freq_bins // 4
        cnn_output_features = 128 * final_freq_bins # 128 channels * 22 bins = 2816
        
        self.bottleneck = nn.Sequential(
            nn.Linear(cnn_output_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.rnn = nn.GRU(
            input_size=256, 
            hidden_size=256, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True,
            dropout=0.2 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 88)      # 512 because bidirectional (256 * 2)
        )

    def forward(self, x):
        x = self.conv_blocks(x) # -> (Batch, 128, 22, Time)
        
        B, C, F, T = x.shape
        x = x.view(B, C * F, T) # -> (Batch, 2816, Time)
        
        x = x.transpose(1, 2)   # -> (Batch, Time, 2816)
        
        x = self.bottleneck(x)  # -> (Batch, Time, 256)
        
        x, _ = self.rnn(x)      # -> (Batch, Time, 512)
        
        x = self.classifier(x)  # -> (Batch, Time, 88)
        
        x = x.transpose(1, 2) 
        
        return x


class ConvStack(nn.Module):

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

class PianoTranscriptionCRNN_v3(nn.Module):
    def __init__(self, freq_bins=88):
        super().__init__()
        
        self.acoustic_model = ConvStack()
        
        final_freq_bins = freq_bins // 4
        cnn_features = 128 * final_freq_bins # 2816 features
        
        self.onset_rnn = nn.GRU(cnn_features, 256, bidirectional=True, batch_first=True)
        self.onset_linear = nn.Linear(512, 88)
        
        self.frame_rnn = nn.GRU(cnn_features + 88, 256, bidirectional=True, batch_first=True)
        self.frame_linear = nn.Linear(512, 88)

    def forward(self, x):
        x = self.acoustic_model(x) 
        B, C, F, T = x.shape
        cnn_out = x.view(B, C * F, T).transpose(1, 2) # -> (Batch, Time, Features)
        
        onset_rnn_out, _ = self.onset_rnn(cnn_out)
        onset_logits = self.onset_linear(onset_rnn_out) # -> (Batch, Time, 88)
        
        onset_probs = torch.sigmoid(onset_logits).detach() 
        
        frame_input = torch.cat([cnn_out, onset_probs], dim=-1) 
        
        frame_rnn_out, _ = self.frame_rnn(frame_input)
        frame_logits = self.frame_linear(frame_rnn_out) # -> (Batch, Time, 88)
        
        return onset_logits.transpose(1, 2), frame_logits.transpose(1, 2)