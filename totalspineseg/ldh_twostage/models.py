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


class SEBlock3d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        r = max(1, int(channels // reduction))
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, r, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(r, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


class ResidualBlock3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int = 1,
        norm: str = "instance",
        use_se: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        def _norm(c: int) -> nn.Module:
            if norm == "instance":
                return nn.InstanceNorm3d(c, affine=True)
            if norm == "group":
                # robust for small batch sizes
                g = 8 if c % 8 == 0 else (4 if c % 4 == 0 else 1)
                return nn.GroupNorm(g, c)
            raise ValueError(f"Unsupported norm={norm!r}. Use: instance|group")

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = _norm(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = _norm(out_ch)
        self.se = SEBlock3d(out_ch) if bool(use_se) else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _norm(out_ch),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.act(self.n1(self.conv1(x)))
        out = self.drop(out)
        out = self.n2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.act(out)


class StageADetectorV2(nn.Module):
    """
    Stronger Stage A detector (residual 3D CNN).
    - Better capacity for harder negatives / partial-coverage cases
    - Designed to be stable with small batch sizes (InstanceNorm/GroupNorm)
    Output: has_LDH logit (B,)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base: int = 32,
        blocks: Tuple[int, int, int] = (2, 2, 2),
        norm: str = "instance",
        use_se: bool = True,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base, affine=True) if norm == "instance" else nn.GroupNorm(8 if base % 8 == 0 else 1, base),
            nn.SiLU(inplace=True),
        )

        def _make_stage(in_ch: int, out_ch: int, n: int, stride: int) -> nn.Sequential:
            layers = [
                ResidualBlock3d(in_ch, out_ch, stride=stride, norm=norm, use_se=use_se, dropout=dropout),
            ]
            for _ in range(int(n) - 1):
                layers.append(ResidualBlock3d(out_ch, out_ch, stride=1, norm=norm, use_se=use_se, dropout=dropout))
            return nn.Sequential(*layers)

        b1, b2, b3 = map(int, blocks)
        self.stage1 = _make_stage(base, base, b1, stride=1)
        self.stage2 = _make_stage(base, base * 2, b2, stride=2)
        self.stage3 = _make_stage(base * 2, base * 4, b3, stride=2)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(base * 4, base * 2),
            nn.SiLU(inplace=True),
            nn.Linear(base * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)


