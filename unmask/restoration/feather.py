from __future__ import annotations

import cv2
import numpy as np


def feather_alpha(mask_255: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return (mask_255.astype(np.float32) / 255.0)[..., None]
    k = max(1, feather_px * 2 + 1)
    blur = cv2.GaussianBlur(mask_255.astype(np.float32), (k, k), 0)
    return np.clip(blur / 255.0, 0.0, 1.0)[..., None]


def blend_restored(
    original_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    mask_255: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    alpha = feather_alpha(mask_255, feather_px)
    out = alpha * restored_bgr.astype(np.float32) + (1.0 - alpha) * original_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
