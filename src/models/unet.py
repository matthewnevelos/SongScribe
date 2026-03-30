import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downscaling with maxpool then residual conv"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upscaling then residual conv with skip connection alignment"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class PianoUNet_v1(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # ENCODER (Downsampling)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = DownBlock(32, 64)   # Freq: 88 -> 44
        self.down2 = DownBlock(64, 128)  # Freq: 44 -> 22
        
        # BOTTLENECK
        self.down3 = DownBlock(128, 256) # Freq: 22 -> 11
        
        # DECODER (Upsampling + Skip Connections)
        self.up1 = UpBlock(256, 128)     # Freq: 11 -> 22
        self.up2 = UpBlock(128, 64)      # Freq: 22 -> 44
        self.up3 = UpBlock(64, 32)       # Freq: 44 -> 88
        
        # OUTPUT
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Input shape: (Batch, 1, 88, Time)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Bottleneck
        x4 = self.down3(x3)
        
        # Decoder (wiring in the skip connections)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output Projection
        logits = self.outc(x) # -> (Batch, 1, 88, Time)
        
        # Squeeze the channel dimension to match your label shape: (Batch, 88, Time)
        logits = logits.squeeze(1) 
        
        return logits
    
    


class ResidualConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU -> Dropout -> Conv2D -> BatchNorm) + Skip Connection"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            # bias=False is used because BatchNorm cancels out the bias anyway
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Spatial dropout drops entire 2D feature maps to prevent co-adaptation
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # The shortcut connection to bypass the block
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Add the original input (shortcut) to the output of the convolutions BEFORE the final ReLU
        return F.relu(self.conv_block(x) + self.shortcut(x), inplace=True)



class PianoUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, dropout_rate=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # ENCODER (Downsampling)
        self.inc = ResidualConv(n_channels, 32, dropout_rate)
        self.down1 = DownBlock(32, 64, dropout_rate)   # Freq: 88 -> 44
        self.down2 = DownBlock(64, 128, dropout_rate)  # Freq: 44 -> 22
        
        # BOTTLENECK (Higher dropout here since it contains the most abstract features)
        self.down3 = DownBlock(128, 256, dropout_rate + 0.1) # Freq: 22 -> 11
        
        # DECODER (Upsampling + Skip Connections)
        self.up1 = UpBlock(256, 128, dropout_rate)     # Freq: 11 -> 22
        self.up2 = UpBlock(128, 64, dropout_rate)      # Freq: 22 -> 44
        self.up3 = UpBlock(64, 32, dropout_rate)       # Freq: 44 -> 88
        
        # OUTPUT
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x) 
        logits = logits.squeeze(1) 
        
        return logits