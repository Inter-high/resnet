"""
This file contains implementations of CIFAR-10 networks using plain blocks and residual blocks.

Author: yumemonzo@gmail.com
Date: 2025-02-24
"""

import torch
import torch.nn as nn
from typing import Type


class PlainBlock(nn.Module):
    """
    A basic block for CIFAR-10 plain networks.
    Consists of two 3x3 convolutions, each followed by Batch Normalization and ReLU.
    No residual shortcut is used.
    """
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class PlainNet(nn.Module):
    """
    A plain network for CIFAR-10.
    Architecture:
      - Initial 3x3 convolution (16 filters)
      - Three groups of blocks:
          * Group 1: 2*n blocks, 16 filters, feature map size 32x32
          * Group 2: 2*n blocks, 32 filters, first block with stride=2 (16x16)
          * Group 3: 2*n blocks, 64 filters, first block with stride=2 (8x8)
      - Global average pooling and a 10-way fully-connected layer.
    Total weighted layers: 6*n+2.
    """
    def __init__(self, block: Type[nn.Module], n: int, num_classes: int = 10) -> None:
        super(PlainNet, self).__init__()
        self.in_planes: int = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, 2 * n, stride=1)
        self.layer2 = self._make_layer(block, 32, 2 * n, stride=2)
        self.layer3 = self._make_layer(block, 64, 2 * n, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResidualBlock(nn.Module):
    """
    A residual block for CIFAR-10.
    Consists of two 3x3 convolutions, each followed by Batch Normalization and ReLU,
    with a shortcut connection that adds the input to the output.
    
    Shortcut types:
      - "A": Uses zero-padding for increasing dimensions with an identity mapping.
      - "B": Uses projection (1x1 conv + BN) only when dimensions increase; otherwise identity.
      - "C": Uses projection for all shortcuts.
    """
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, shortcut_type: str = "B") -> None:
        super(ResidualBlock, self).__init__()
        self.shortcut_type = shortcut_type
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes: int = planes * self.expansion

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if shortcut_type == "C":
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_planes)
            )
        elif shortcut_type == "B":
            if stride != 1 or in_planes != self.out_planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.out_planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.out_planes)
                )
            else:
                self.shortcut = nn.Identity()
        elif shortcut_type == "A":
            if stride != 1 or in_planes != self.out_planes:
                self.need_pad = True
            else:
                self.need_pad = False
                self.shortcut = nn.Identity()
        else:
            raise ValueError("Invalid shortcut_type. Choose from 'A', 'B', or 'C'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut_type in ["B", "C"]:
            shortcut = self.shortcut(x)
        elif self.shortcut_type == "A":
            if self.need_pad:
                if self.stride != 1:
                    x = x[:, :, ::self.stride, ::self.stride]
                ch_pad = self.out_planes - self.in_planes
                padding = torch.zeros(x.size(0), ch_pad, x.size(2), x.size(3),
                                      device=x.device, dtype=x.dtype)
                shortcut = torch.cat([x, padding], dim=1)
            else:
                shortcut = x
        else:
            shortcut = x
        
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    A ResNet for CIFAR-10.
    Architecture:
      - Initial 3x3 convolution (16 filters)
      - Three groups of residual blocks:
          * Group 1: 2*n blocks, 16 filters, feature map size 32x32
          * Group 2: 2*n blocks, 32 filters, first block with stride=2 (16x16)
          * Group 3: 2*n blocks, 64 filters, first block with stride=2 (8x8)
      - Global average pooling and a 10-way fully-connected layer.
    Total weighted layers: 6*n+2.
    
    Shortcut type can be "A", "B", or "C" (default "B").
    """
    def __init__(self, block: Type[nn.Module], n: int, num_classes: int = 10, shortcut_type: str = "B") -> None:
        super(ResNet, self).__init__()
        self.in_planes: int = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, 2 * n, stride=1, shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(block, 32, 2 * n, stride=2, shortcut_type=shortcut_type)
        self.layer3 = self._make_layer(block, 64, 2 * n, stride=2, shortcut_type=shortcut_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int, shortcut_type: str) -> nn.Sequential:
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut_type))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut_type=shortcut_type))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

def get_plain_network(n: int, num_classes: int = 10) -> PlainNet:
    """
    Creates a plain network for CIFAR-10 with 6*n+2 layers.

    Args:
        n (int): Determines the depth of the network.
        num_classes (int): Number of output classes.

    Returns:
        PlainNet: An instance of the plain network.
    """
    return PlainNet(PlainBlock, n, num_classes=num_classes)


def get_resnet(n: int, num_classes: int = 10, shortcut_type: str = "B") -> ResNet:
    """
    Creates a ResNet for CIFAR-10 with 6*n+2 layers.

    Args:
        n (int): Determines the depth of the network.
        num_classes (int): Number of output classes.
        shortcut_type (str): Type of shortcut to use ("A", "B", or "C").

    Returns:
        ResNet: An instance of the ResNet.
    """
    return ResNet(ResidualBlock, n, num_classes=num_classes, shortcut_type=shortcut_type)
