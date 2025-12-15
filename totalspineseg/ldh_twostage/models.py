from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for ROI LDH segmentation.
    Outputs:
      - mask_logits: (B,1,Z,Y,X)
      - sdm_pred:    (B,1,Z,Y,X)
    """

    def __init__(self, in_channels: int = 3, base: int = 24):
        super().__init__()
        self.enc1 = ConvBlock3d(in_channels, base)
        self.down1 = nn.Conv3d(base, base * 2, kernel_size=2, stride=2)
        self.enc2 = ConvBlock3d(base * 2, base * 2)
        self.down2 = nn.Conv3d(base * 2, base * 4, kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3d(base * 4, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(base * 2, base)

        self.head_mask = nn.Conv3d(base, 1, kernel_size=1)
        self.head_sdm = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head_mask(d1), self.head_sdm(d1)


class StageADetector(nn.Module):
    """
    Disc-level LDH detection. Prior is provided by disc mask + disc index channel.
    Output: has_LDH logit.
    """

    def __init__(self, in_channels: int = 3, feat: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(feat, feat * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(feat * 2, feat * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat * 4, feat * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.fc(f).squeeze(1)


