import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Standard residual block without spatial downsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DopplerHead(nn.Module):
    """
    3D convolutional head that collapses the Doppler dimension.

    Input:  (B, C, D, H, W)  — C channels (power + elevation), D Doppler bins
    Output: (B, 64, H, W)    — spatial feature map for the 2D backbone
    """

    def __init__(self, in_channels=2, out_channels=64, doppler_depth=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16,          kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False)
        self.bn1   = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16,          out_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        # Collapse the Doppler dimension entirely
        self.pool  = nn.MaxPool3d(kernel_size=(doppler_depth, 1, 1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x.squeeze(2)   # (B, out_channels, H, W)


class RadarResNet(nn.Module):
    """
    Radar 3D occupancy network.

    Architecture:
      DopplerHead  — 3D conv + MaxPool collapses (B,2,128,256,256) → (B,64,256,256)
      4× BasicBlock layers (ResNet-18 style, no spatial downsampling)
      1×1 Conv output head → (B, num_classes, 256, 256)

    Output logits are thresholded at 0.5 after sigmoid during inference.
    """

    def __init__(self, in_channels=2, num_classes=64, doppler_depth=128):
        super().__init__()
        self.head   = DopplerHead(in_channels=in_channels, out_channels=64, doppler_depth=doppler_depth)
        self._ch    = 64
        self.layer1 = self._make_layer(64,  2)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4      = self._make_layer(512, 2)
        self.output_head = nn.Conv2d(512, num_classes, kernel_size=1)

    def _make_layer(self, out_channels, n_blocks):
        layers = []
        for _ in range(n_blocks):
            layers.append(BasicBlock(self._ch, out_channels))
            self._ch = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)     # (B, 64,  256, 256)
        x = self.layer1(x)   # (B, 64,  256, 256)
        x = self.layer2(x)   # (B, 128, 256, 256)
        x = self.layer3(x)   # (B, 256, 256, 256)
        x = self.layer4(x)   # (B, 512, 256, 256)
        return self.output_head(x)   # (B, num_classes, 256, 256)
