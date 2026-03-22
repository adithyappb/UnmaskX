from __future__ import annotations

import numpy as np

from unmask.config import Settings
from unmask.detection.mediapipe import FaceRegionDetector
from unmask.restoration.registry import restore_lower_face


class RestorationPipeline:
    """Detect lower-face region, then restore via registered backends (OpenCV / U-Net / GAN plugin)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._detector = FaceRegionDetector(
            model_path=self.settings.landmarker_model_path,
            max_faces=self.settings.max_faces,
            min_face_detection_confidence=self.settings.min_detection_confidence,
            min_face_presence_confidence=self.settings.min_face_presence_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> RestorationPipeline:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def process_bgr(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        _, mask = self._detector.detect_mask_region(frame_bgr, self.settings.mask_mode)
        if mask is None:
            return frame_bgr, False

        out = restore_lower_face(frame_bgr, mask, self.settings)
        return out, True
