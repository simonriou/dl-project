import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMaskModel(nn.Module):
    def __init__(self):
        super(CNNMaskModel, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Decoder / upsampling
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(32)

        # Final conv
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        # Decoder
        x = self.up1(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.up2(x)
        x = F.relu(self.bn8(self.conv8(x)))

        # Crop 3 freq bins to match original shape (513, 188)
        x = x[:, :, :513, :]

        x = torch.sigmoid(self.conv_out(x))
        return x

def build_cnn_mask_model(input_shape=None):
    # input_shape not strictly needed in PyTorch
    return CNNMaskModel()