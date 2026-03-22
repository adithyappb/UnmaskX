from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration: face region detection + lower-face restoration."""

    landmarker_model_path: Path = field(default_factory=lambda: _assets_dir() / "face_landmarker.task")
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    mask_mode: Literal["mesh", "bbox"] = "mesh"

    # restoration_backend: auto picks unet if weights exist, else opencv. "gan" is pluggable (see docs/PLUGINS.md).
    restoration_backend: Literal["auto", "opencv", "unet", "gan"] = "auto"
    opencv_algorithm: Literal["ns", "telea"] = "ns"
    opencv_inpaint_radius: int = 7
    opencv_multiscale: bool = True

    unet_weights: Path | None = None
    unet_input_size: int = 256
    unet_base_channels: int = 48

    feather_px: int = 22

    # Edge-preserving pass on the filled region (helps vs. blurry inpainting; tiny CPU cost)
    bilateral_refinement: bool = True
    bilateral_d: int = 7
    bilateral_sigma_color: float = 55.0
    bilateral_sigma_space: float = 55.0

    assets_dir: Path = field(default_factory=_assets_dir)


def resolve_unet_weights(s: Settings) -> Path | None:
    p = s.unet_weights if s.unet_weights is not None else s.assets_dir / "unmask_unet.pth"
    return p if p.is_file() else None


def effective_restoration_id(s: Settings) -> str:
    if s.restoration_backend == "unet":
        return "unet"
    if s.restoration_backend == "opencv":
        return "opencv"
    if s.restoration_backend == "gan":
        return "gan"
    return "unet" if resolve_unet_weights(s) is not None else "opencv"


