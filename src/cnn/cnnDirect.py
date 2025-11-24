import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetSpectrogram(nn.Module):
    """
    U-Net style CNN for clean spectrogram regression.
    Input:  (B, 1, F, T)
    Output: (B, 1, F, T) clean magnitude spectrogram
    """

    def __init__(self):
        super().__init__()

        # -----------------------
        # Encoder
        # -----------------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # -----------------------
        # Decoder
        # -----------------------
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Final projection (1 channel)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

        # Use Softplus to keep output positive without saturation
        self.out_act = nn.Softplus()


    def forward(self, x):
        # -----------------------
        # Encoder with skip connections
        # -----------------------
        e1 = self.enc1(x)           # (B, 32, F, T)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)          # (B, 64, F/2, T/2)
        p2 = self.pool2(e2)

        b  = self.bottleneck(p2)    # (B, 128, F/4, T/4)

        # -----------------------
        # Decoder with concatenation of skips
        # -----------------------
        d1 = self.up1(b)            # (B, 64, F/2, T/2)

        # Adjust shape if needed (due to ceil_mode in pooling)
        if d1.shape[-2:] != e2.shape[-2:]:
            d1 = F.interpolate(d1, size=e2.shape[-2:], mode="nearest")

        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)           # (B, 32, F, T)

        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")

        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        # Final output
        out = self.out_conv(d2)
        out = self.out_act(out)     # non-negative clean spectrogram

        # If your STFT uses F=513, crop if needed
        out = out[:, :, :513, :]

        return out
    
def build_cnn_direct_model(input_shape=None):
    # input_shape not strictly needed in PyTorch
    return UNetSpectrogram()