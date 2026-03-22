"""Lightweight edge-preserving refinement on the restored region (reduces mushy inpainting)."""

from __future__ import annotations

import cv2
import numpy as np

from unmask.restoration.feather import feather_alpha


def bilateral_inpaint_region(
    blended_bgr: np.ndarray,
    mask_255: np.ndarray,
    feather_px: int,
    d: int = 7,
    sigma_color: float = 55.0,
    sigma_space: float = 55.0,
) -> np.ndarray:
    """
    Apply bilateral filter to the full frame, then blend so only the (soft) mask area
    picks up edge-preserving smoothing; keeps skin outside the hole stable.
    """
    if d < 1:
        return blended_bgr
    d = d if d % 2 == 1 else d + 1
    smoothed = cv2.bilateralFilter(blended_bgr, d, sigma_color, sigma_space)
    m = feather_alpha(mask_255, feather_px)
    return (m * smoothed.astype(np.float32) + (1.0 - m) * blended_bgr.astype(np.float32)).astype(np.uint8)
