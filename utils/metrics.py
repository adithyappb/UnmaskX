from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse_val = mse(pred, target)
    if mse_val == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse_val)))


def evaluate_inpainting(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"MSE": mse(pred, target), "PSNR": psnr(pred, target)}


def masked_psnr_numpy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """pred, target: H,W,3 or 3,H,W in [0,255]; mask: H,W in [0,1]."""
    if pred.ndim == 3 and pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
    w = np.clip(mask, 0, 1)
    if w.sum() < 1:
        return 0.0
    diff = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
    mse_val = (diff * w[..., None]).sum() / (w.sum() * pred.shape[-1] + 1e-6)
    return float(10 * np.log10(255.0**2 / (mse_val + 1e-6)))


def masked_batch_metrics(
    pred: "torch.Tensor",
    tgt: "torch.Tensor",
    mask: "torch.Tensor",
) -> tuple[float, float]:
    """pred, tgt: N,3,H,W in [-1,1]; mask: N,1,H,W in [0,1]. Mean masked PSNR + mean SSIM."""
    pred = pred.detach().cpu().numpy()
    tgt = tgt.detach().cpu().numpy()
    m = mask.detach().cpu().numpy()
    psnrs: list[float] = []
    ssims: list[float] = []
    for i in range(pred.shape[0]):
        p = np.clip((pred[i] + 1.0) * 0.5, 0.0, 1.0) * 255.0
        t = np.clip((tgt[i] + 1.0) * 0.5, 0.0, 1.0) * 255.0
        w = np.clip(m[i, 0], 0.0, 1.0)
        psnrs.append(masked_psnr_numpy(p, t, w))
        if ssim_fn is not None:
            pa = np.transpose(p, (1, 2, 0)) / 255.0
            ta = np.transpose(t, (1, 2, 0)) / 255.0
            try:
                s = ssim_fn(ta, pa, data_range=1.0, channel_axis=-1)
            except TypeError:
                s = ssim_fn(ta, pa, multichannel=True, data_range=1.0)
            ssims.append(float(s))
    return float(np.mean(psnrs)), float(np.mean(ssims)) if ssims else 0.0


def composite_quality_index(
    psnr: float,
    ssim: float,
    lpips: float | None = None,
) -> float:
    """
    Single 0–100 score for comparing runs over time (not identity accuracy).

    - PSNR is mapped from roughly 10–40 dB into [0, 1].
    - SSIM is already in [0, 1].
    - If LPIPS is provided (lower is better), it is folded in as (1 − min(LPIPS, 1)).
    """
    n_psnr = float(np.clip((psnr - 10.0) / 30.0, 0.0, 1.0))
    n_ssim = float(np.clip(ssim, 0.0, 1.0))
    if lpips is not None and np.isfinite(lpips):
        n_lp = float(np.clip(1.0 - min(float(lpips), 1.0), 0.0, 1.0))
        return float(100.0 * (0.30 * n_psnr + 0.45 * n_ssim + 0.25 * n_lp))
    return float(100.0 * (0.45 * n_psnr + 0.55 * n_ssim))


def append_metrics_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
