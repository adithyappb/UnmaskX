"""Mask-conditioned U-Net (train with `python -m training.train_unet`)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _gn_groups(c: int) -> int:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(c_out), c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(c_out), c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskInpaintUNet(nn.Module):
    """4-channel in (masked RGB + mask), 3-channel RGB out in [-1, 1].

    Default forward: **masked residual** — returns base + mask * tanh(delta) so visible pixels
    match the input (no drift) and the network only predicts inside the hole (avoids gray blobs).
    Set legacy_output=True for checkpoints trained with the old full-frame tanh output.
    """

    def __init__(self, base: int = 48, legacy_output: bool = False) -> None:
        super().__init__()
        self.legacy_output = legacy_output
        self.enc1 = ConvBlock(4, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 4, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 2 + base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base + base * 2, base)
        self.up0 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec0 = ConvBlock(base + base, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, e1], dim=1))
        delta = torch.tanh(self.out(d0))
        if self.legacy_output:
            return delta
        base = x[:, :3]
        m = x[:, 3:4]
        return base + m * delta


