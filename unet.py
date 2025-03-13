import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    # Two convolutional layers (3×3 kernel, same padding).
    # Batch normalization (optional but stabilizes training).
    # ReLU activation after each convolution.
    # used at each step of the UNet on both sides

    def __init__(self, input_channels, output_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    # 4 feature extraction and downsampling steps

    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()
        
        self.enc64 = ConvBlock(in_channels, 64)
        self.enc128 = ConvBlock(64, 128)
        self.enc256 = ConvBlock(128, 256)
        self.enc512 = ConvBlock(256, 512)

        self.pool_by_half = nn.MaxPool2d(kernel_size=2, stride=2) # halves the dimensions

    def forward(self, x):
        x1 = self.enc64(x)  # 64 channels -> output 256x256x64
        x2 = self.pool_by_half(x1) # halve dimensions -> output 128x128x64

        x2 = self.enc128(x2) # 128 channels -> output 128x128x128
        x3 = self.pool_by_half(x2) # halve dimensions -> output 64x64x128

        x3 = self.enc256(x3) # 256 channels -> output 64x64x256
        x4 = self.pool_by_half(x3) # halve dimensions -> output 32x32x256

        x4 = self.enc512(x4) # 512 channels -> output 32x32x512
        x5 = self.pool_by_half(x4) # halve dimensions -> output 16x16x512

        return x1, x2, x3, x4, x5



class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super(UNetDecoder, self).__init__()

        self.upconv256 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.dec256 = ConvBlock(512, 256)  

        self.upconv128 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec128 = ConvBlock(256, 128)

        self.upconv64 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec64 = ConvBlock(128, 64)

        self.upconv32 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec32 = ConvBlock(64, 32)

        self.final_conv1 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x5, x4, x3, x2, x1):
        x = self.upconv256(x5) # upsample 16x16x512 -> 32x32x256

        x = torch.cat([x, x4], dim=1)  # Concatenate with skip connection, now 256+256 = 512 channels, dimensions 32x32x512
        x = self.dec256(x)  # 32x32x512 -> 32x32x256 output

        x = self.upconv128(x)  # Upsample to 64x64x128
        x = torch.cat([x, x3], dim=1)  # Concatenate, now 64x64x256
        x = self.dec128(x) # 64x64x256 -> 64x64x128 output

        x = self.upconv64(x)  # Upsample to 128x128x128
        x = torch.cat([x, x2], dim=1)  # Concatenate, now 128x128x256
        x = self.dec64(x) # 128x128x256 -> 128x128x64 output

        x = self.upconv32(x)  # Upsample to 256x256x32
        x = torch.cat([x, x1], dim=1) #  Concatenate, now 256x256x64
        x = self.dec32(x) # 256x256x64 -> 256x256x32

        x = self.final_conv1(x)  # Convert to 1-channel output (segmentation map) 256x256x32 -> 256x256x1
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)  # Get encoder outputs

        x = self.decoder(x5, x4, x3, x2, x1)  # Pass through decoder

        return x
