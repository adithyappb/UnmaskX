from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from unmask.config import Settings


@runtime_checkable
class FaceRestorer(Protocol):
    """Pluggable lower-face restoration (inpainting / generative). Implement and register in the registry."""

    name: str

    def restore(self, image_bgr: np.ndarray, mask_255: np.ndarray, settings: "Settings") -> np.ndarray:
        """Return full-frame BGR uint8 with the masked region filled."""
