"""Fill masked regions with visible-region mean color (avoids gray 0.5 collapse in inpainting)."""

from __future__ import annotations

import numpy as np


def hole_fill_rgb01(rgb: np.ndarray, mask_01: np.ndarray) -> np.ndarray:
    """
    rgb: H,W,3 float32 in [0,1]
    mask_01: H,W float in [0,1], 1 = unknown / hole to inpaint
    Returns composited known view (visible pixels unchanged, hole filled with per-image mean skin tone).
    """
    inv = np.clip(1.0 - mask_01, 0.0, 1.0)
    s = float(inv.sum()) + 1e-6
    mean = (rgb.astype(np.float64) * inv[..., None]).sum(axis=(0, 1)) / s
    mean = np.clip(mean, 0.0, 1.0).astype(np.float32)
    if not np.all(np.isfinite(mean)):
        mean = np.full(3, 0.5, dtype=np.float32)
    return rgb * (1.0 - mask_01[..., None]) + mean * mask_01[..., None]
