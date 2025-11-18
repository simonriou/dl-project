import torch
import torch.nn as nn
import torch.nn.functional as F

def match_size(tensor, target):
    """
    Resize tensor to match target's spatial size using interpolation.
    """
    _, _, th, tw = target.shape
    return F.interpolate(tensor, size=(th, tw), mode='bilinear', align_corners=False)

class CNNMaskModel(nn.Module): # Model 1-4
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

class CNNMaskUNet(nn.Module): # Model 5
    def __init__(self):
        super(CNNMaskUNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)  # concat skip
        self.bn7 = nn.BatchNorm2d(64)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)  # concat skip
        self.bn8 = nn.BatchNorm2d(32)

        # Final conv
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))  # input is previous output
        p1 = self.pool1(x1)

        x2 = F.relu(self.bn3(self.conv3(p1)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        p2 = self.pool2(x2)

        x3 = F.relu(self.bn5(self.conv5(p2)))
        x3 = F.relu(self.bn6(self.conv6(x3)))

        # Decoder
        u1 = self.up1(x3)
        x2_matched = match_size(x2, u1)   # ensure same height and width
        u1 = torch.cat([u1, x2_matched], dim=1)
        u1 = F.relu(self.bn7(self.conv7(u1)))

        u2 = self.up2(u1)
        x1_matched = match_size(x1, u2)
        u2 = torch.cat([u2, x1_matched], dim=1)
        u2 = F.relu(self.bn8(self.conv8(u2)))

        # Crop freq bins to match original shape
        u2 = u2[:, :, :513, :]

        out = torch.sigmoid(self.conv_out(u2))
        return out

def build_cnn_mask_model(input_shape=None):
    # input_shape not strictly needed in PyTorch
    return CNNMaskUNet()