from __future__ import annotations

import cv2
import numpy as np
import torch

from unmask.config import Settings, resolve_unet_weights
from unmask.nn.unet import MaskInpaintUNet
from unmask.hole_fill import hole_fill_rgb01


class PyTorchUNetRestorer:
    name = "unet"

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def _load(self, settings: Settings) -> None:
        if self._model is not None:
            return
        path = resolve_unet_weights(settings)
        if path is None:
            raise FileNotFoundError("No U-Net weights found (see training.train_unet).")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            raw = torch.load(path, map_location=self._device, weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location=self._device)

        base = settings.unet_base_channels
        sd = raw
        legacy_output = True
        if isinstance(raw, dict) and "model_state_dict" in raw:
            sd = raw["model_state_dict"]
            base = raw.get("config", {}).get("base", base)
            if raw.get("training_config") and isinstance(raw["training_config"], dict):
                base = raw["training_config"].get("base", base)
            if raw.get("inpaint_forward") == "masked_residual":
                legacy_output = False
        elif isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]

        net = MaskInpaintUNet(base=base, legacy_output=legacy_output).to(self._device)
        net.load_state_dict(sd, strict=True)
        net.eval()
        self._model = net

    def restore(self, image_bgr: np.ndarray, mask_255: np.ndarray, settings: Settings) -> np.ndarray:
        self._load(settings)
        assert self._model is not None and self._device is not None

        h, w = image_bgr.shape[:2]
        size = settings.unet_input_size
        img_s = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(mask_255, (size, size), interpolation=cv2.INTER_LINEAR)
        m_s = cv2.GaussianBlur(m_s.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)

        rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m = np.clip(m_s.astype(np.float32) / 255.0, 0.0, 1.0)
        known = hole_fill_rgb01(rgb, m)

        rgb_t = torch.from_numpy(known).permute(2, 0, 1).unsqueeze(0).to(self._device)
        rgb_t = rgb_t * 2.0 - 1.0
        m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(self._device)
        xt = torch.cat([rgb_t, m_t], dim=1)

        with torch.no_grad():
            y = self._model(xt)
        out = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out + 1.0) * 0.5
        out = np.clip(out, 0.0, 1.0)
        out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
