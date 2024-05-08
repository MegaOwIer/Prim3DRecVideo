import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from utils import WNConv2d


class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = x + skip

        return x


class ResidualBlockMLP(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockMLP, self).__init__()

        self.in_norm = nn.BatchNorm1d(in_channels)
        self.in_linear = nn.Linear(in_channels, out_channels)

        self.out_norm = nn.BatchNorm1d(out_channels)
        self.out_linear = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        in_x = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_linear(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_linear(x)

        x = x + in_x

        return x

