from __future__ import annotations

import cv2
import numpy as np

from unmask.config import Settings


def _single(
    image_bgr: np.ndarray,
    mask_255: np.ndarray,
    radius: int,
    algorithm: str,
) -> np.ndarray:
    flag = cv2.INPAINT_NS if algorithm == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(image_bgr, mask_255, radius, flag)


def _multiscale_ns(
    image_bgr: np.ndarray,
    mask_255: np.ndarray,
    radius: int,
    algorithm: str,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if min(h, w) < 128:
        return _single(image_bgr, mask_255, radius, algorithm)

    h2, w2 = max(1, h // 2), max(1, w // 2)
    small = cv2.resize(image_bgr, (w2, h2), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask_255, (w2, h2), interpolation=cv2.INTER_NEAREST)
    r_small = max(2, radius // 2)
    coarse = _single(small, msmall, r_small, algorithm)
    coarse_up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    fine = _single(image_bgr, mask_255, radius, algorithm)

    m = (mask_255.astype(np.float32) / 255.0)[..., None]
    edge = cv2.GaussianBlur(mask_255.astype(np.float32), (0, 0), sigmaX=5, sigmaY=5)
    edge = np.clip(edge / 255.0, 0, 1.0)[..., None]
    interior = np.clip(m - edge * 0.85, 0.0, 1.0)
    blend = fine.astype(np.float32) * (1.0 - interior) + coarse_up.astype(np.float32) * interior
    return np.clip(blend, 0, 255).astype(np.uint8)


class OpenCVRestorer:
    name = "opencv"

    def restore(self, image_bgr: np.ndarray, mask_255: np.ndarray, settings: Settings) -> np.ndarray:
        if settings.opencv_multiscale:
            return _multiscale_ns(
                image_bgr,
                mask_255,
                settings.opencv_inpaint_radius,
                settings.opencv_algorithm,
            )
        return _single(
            image_bgr,
            mask_255,
            settings.opencv_inpaint_radius,
            settings.opencv_algorithm,
        )
