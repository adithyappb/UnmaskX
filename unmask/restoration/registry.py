from __future__ import annotations

import importlib
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unmask.config import Settings

_restorers: dict[str, type] = {}
_instances: dict[str, object] = {}


def register(name: str, cls: type) -> None:
    _restorers[name] = cls


def list_names() -> list[str]:
    _ensure_builtins()
    return sorted(_restorers.keys())


def get(name: str):
    _ensure_builtins()
    if name not in _restorers:
        raise KeyError(f"Unknown restorer {name!r}. Registered: {list_names()}")
    if name not in _instances:
        _instances[name] = _restorers[name]()
    return _instances[name]


def _ensure_builtins() -> None:
    if _restorers:
        return
    from unmask.restoration.gan_placeholder import GANPlaceholderRestorer
    from unmask.restoration.opencv import OpenCVRestorer
    from unmask.restoration.pytorch_unet import PyTorchUNetRestorer

    register("opencv", OpenCVRestorer)
    register("unet", PyTorchUNetRestorer)
    register("gan", GANPlaceholderRestorer)
    _maybe_override_gan_from_env()


def _maybe_override_gan_from_env() -> None:
    spec = os.environ.get("UNMASK_GAN_CLASS", "").strip()
    if not spec:
        return
    try:
        mod_name, _, cls_name = spec.partition(":")
        if not cls_name:
            raise ValueError("Use module.path:ClassName")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        _restorers["gan"] = cls
        _instances.pop("gan", None)
    except Exception as e:
        warnings.warn(f"Could not load UNMASK_GAN_CLASS={spec!r}: {e}", stacklevel=2)


def restore_lower_face(image_bgr: np.ndarray, mask_255: np.ndarray, settings: "Settings") -> np.ndarray:
    from unmask.config import Settings, effective_restoration_id
    from unmask.restoration.feather import blend_restored
    from unmask.restoration.refine import bilateral_inpaint_region

    if not isinstance(settings, Settings):
        raise TypeError("settings must be Settings")

    rid = effective_restoration_id(settings)
    try:
        raw = get(rid).restore(image_bgr, mask_255, settings)
    except Exception as e:
        if rid != "opencv":
            warnings.warn(f"Restorer {rid!r} failed ({e!s}); falling back to opencv.", UserWarning, stacklevel=2)
            raw = get("opencv").restore(image_bgr, mask_255, settings)
        else:
            raise

    blended = blend_restored(image_bgr, raw, mask_255, settings.feather_px)
    if settings.bilateral_refinement:
        blended = bilateral_inpaint_region(
            blended,
            mask_255,
            settings.feather_px,
            d=settings.bilateral_d,
            sigma_color=settings.bilateral_sigma_color,
            sigma_space=settings.bilateral_sigma_space,
        )
    return blended
