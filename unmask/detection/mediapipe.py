from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover
    raise ImportError("Install mediapipe: pip install mediapipe") from e

_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

_MASK_MESH_INDICES = tuple(
    dict.fromkeys(
        (
            6,
            19,
            20,
            94,
            2,
            164,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            200,
            199,
            175,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
        )
    )
)


def ensure_landmarker_model(path: Path, url: str = _DEFAULT_MODEL_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return path


class FaceRegionDetector:
    """MediaPipe Face Landmarker + binary mask for the lower-face (mask) region."""

    def __init__(
        self,
        model_path: Path,
        max_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        ensure_landmarker_model(model_path)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceRegionDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def detect_mask_region(
        self,
        image_bgr: np.ndarray,
        mask_mode: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if image_bgr is None or image_bgr.size == 0:
            return None, None

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None, None

        landmarks = result.face_landmarks[0]
        if mask_mode == "bbox":
            mask = self._bbox_lower_face_mask(landmarks, w, h)
        else:
            mask = self._mesh_convex_hull_mask(landmarks, w, h)

        if mask is None or not np.any(mask):
            return None, None

        return rgb, mask

    def _mesh_convex_hull_mask(self, landmarks, w: int, h: int) -> np.ndarray:
        pts = []
        for idx in _MASK_MESH_INDICES:
            lm = landmarks[idx]
            pts.append([lm.x * w, lm.y * h])
        pts = np.array(pts, dtype=np.float32)
        hull = cv2.convexHull(pts.astype(np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def _bbox_lower_face_mask(self, landmarks, w: int, h: int) -> np.ndarray:
        xs = [landmarks[i].x * w for i in (234, 454, 152, 1)]
        ys = [landmarks[i].y * h for i in (234, 454, 152, 1)]
        x1, x2 = int(min(xs)), int(max(xs))
        y1 = int(landmarks[1].y * h + 0.02 * h)
        y2 = int(landmarks[152].y * h + 0.02 * h)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask


