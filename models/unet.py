import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """Simplified U-Net for feature inversion."""
    def __init__(self, input_channels, output_channels, feature_size=16):
        super().__init__()
        self.feature_size = feature_size
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder with proper channel handling for skip connections
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)
        
        # Skip connection processing layers
        self.skip4 = self._conv_block(1024, 512)  # 512 (decoder) + 512 (encoder) = 1024
        self.skip3 = self._conv_block(512, 256)   # 256 (decoder) + 256 (encoder) = 512
        self.skip2 = self._conv_block(256, 128)   # 128 (decoder) + 128 (encoder) = 256
        self.skip1 = self._conv_block(128, 64)    # 64 (decoder) + 64 (encoder) = 128
        
        # Final layer
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Reshape features to spatial format if needed
        if len(x.shape) == 4 and x.shape[2] == x.shape[3] == self.feature_size:
            # Already in (B, C, H, W) format
            pass
        else:
            # Reshape from (B, H, W, C) to (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        
        # Encoder path
        e1 = self.enc1(x)           # (B, 64, feature_size, feature_size)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder path with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.skip4(d4)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.skip3(d3)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.skip2(d2)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.skip1(d1)
        
        # Final output
        output = self.final(d1)
        return torch.tanh(output)  # Normalize to [-1, 1]