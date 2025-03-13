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
        
        self.enc32 = ConvBlock(in_channels, 32)
        self.enc64 = ConvBlock(32, 64)
        self.enc128 = ConvBlock(64, 128)
        self.enc256 = ConvBlock(128, 256)
        self.enc512 = ConvBlock(256, 512)

        self.pool_by_half = nn.MaxPool2d(kernel_size=2, stride=2) # halves the dimensions

    def forward(self, x):
        encoding_32 = self.enc32(x)  # 64 channels -> output 256x256x32
        pooled_32 = self.pool_by_half(encoding_32) # halve dimensions -> output 128x128x32

        encoding_64 = self.enc64(pooled_32)  # 64 channels -> output 128x128x64
        pooled_64 = self.pool_by_half(encoding_64) # halve dimensions -> output 64x64x64

        encoding_128 = self.enc128(pooled_64) # 128 channels -> output 64x64x128
        pooled_128 = self.pool_by_half(encoding_128) # halve dimensions -> output 32x32x128

        encoding_256 = self.enc256(pooled_128) # 256 channels -> output 32x32x256
        pooled_256 = self.pool_by_half(encoding_256) # halve dimensions -> output 16x16x256

        encoding_512 = self.enc512(pooled_256) # 512 channels -> output 16x16x512

        return encoding_32, encoding_64, encoding_128, encoding_256, encoding_512


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

    def forward(self, encoding_512, encoding_256, encoding_128, encoding_64, encoding_32):
        upsampled256 = self.upconv256(encoding_512) # upsample 16x16x512 -> 32x32x256
        concat256 = torch.cat([upsampled256, encoding_256], dim=1)  # Concatenate with skip connection, now 256+256 = 512 channels, dimensions 32x32x512
        decoding256 = self.dec256(concat256)  # 32x32x512 -> 32x32x256 output

        upsampled128 = self.upconv128(decoding256)  # Upsample to 64x64x128
        concat128 = torch.cat([upsampled128, encoding_128], dim=1)  # Concatenate, now 64x64x256
        decoding128 = self.dec128(concat128) # 64x64x256 -> 64x64x128 output

        upsampled64 = self.upconv64(decoding128)  # Upsample to 128x128x64
        concat64 = torch.cat([upsampled64, encoding_64], dim=1)  # Concatenate, now 128x128x256
        decoding64 = self.dec64(concat64) # 128x128x256 -> 128x128x64 output

        upsampled32 = self.upconv32(decoding64)  # Upsample to 256x256x32
        concat32 = torch.cat([upsampled32, encoding_32], dim=1) #  Concatenate, now 256x256x64
        decoding32 = self.dec32(concat32) # 256x256x64 -> 256x256x32

        result = self.final_conv1(decoding32)  # Convert to 1-channel output (segmentation map) 256x256x32 -> 256x256x1
        return result


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)  # Get encoder outputs

        x = self.decoder(x5, x4, x3, x2, x1)  # Pass through decoder

        return x
