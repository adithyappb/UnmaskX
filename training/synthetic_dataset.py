"""
Synthetic lower-face masks for training mask-conditioned inpainting.

Use **uncovered face** images only: the model learns to reconstruct the lower face from a
random synthetic mask. Put real **wearing-mask** photos in the same folder only if you
exclude them from training (see --exclude / auto-exclude rules).
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from unmask.hole_fill import hole_fill_rgb01

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Subfolders excluded from training scans (holdout / validation / unmerged archives).
_EXCLUDED_SUBDIR_NAMES = frozenset({"test", "val", "validation", "holdout", "exclude"})


def _manual_exclude(path: Path, extra_substrings: tuple[str, ...]) -> bool:
    """Explicit --exclude substrings (matched against full filename, case-insensitive)."""
    name = path.name.lower()
    for x in extra_substrings:
        x = x.strip().lower()
        if x and x in name:
            return True
    return False


def _heuristic_is_masked_reference(path: Path) -> bool:
    """
    Skip photos that already show a real surgical mask (not used for synthetic-mask training).
    Filenames with 'unmasked' are never skipped by this heuristic.
    """
    stem = path.stem.lower()
    name = path.name.lower()
    if "unmasked" in stem:
        return False
    if stem in ("mask", "masked", "with_mask", "mask_only", "wearing_mask"):
        return True
    if "with_mask" in stem or "with-mask" in name or "with mask" in name:
        return True
    if "_masked" in stem or stem.startswith("masked_"):
        return True
    return False


def _read_train_list(path: Path) -> list[str]:
    lines: list[str] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def collect_face_images(
    root: str | Path,
    extra_exclude: tuple[str, ...] = (),
    auto_exclude_masked: bool = True,
    train_list_path: Path | None = None,
    only_names: tuple[str, ...] | None = None,
) -> list[Path]:
    root = Path(root)
    tlp = train_list_path
    if tlp is None:
        cand = root / "train_list.txt"
        tlp = cand if cand.is_file() else None

    names: list[str] | None = None
    if tlp is not None and tlp.is_file():
        names = _read_train_list(tlp)
        if not names:
            names = None

    def _keep(p: Path) -> bool:
        if _manual_exclude(p, extra_exclude):
            return False
        if auto_exclude_masked and _heuristic_is_masked_reference(p):
            return False
        if only_names:
            if p.name not in only_names and p.stem not in only_names:
                return False
        return True

    def _path_ok(p: Path) -> bool:
        try:
            rel = p.relative_to(root.resolve())
        except ValueError:
            return True
        for part in rel.parts:
            pl = part.lower()
            if pl in _EXCLUDED_SUBDIR_NAMES or pl.startswith("archive_"):
                return False
        return True

    if names is not None:
        out: list[Path] = []
        for n in names:
            p = (root / n).resolve()
            if not p.is_file():
                continue
            if _keep(p) and _path_ok(p):
                out.append(p)
        return sorted(set(out))

    raw: list[Path] = []
    for ext in _IMG_EXT:
        raw.extend(root.rglob(f"*{ext}"))
    out2: list[Path] = []
    for p in sorted({q.resolve() for q in raw}):
        if _keep(p) and _path_ok(p):
            out2.append(p)
    return out2


def random_surgical_mask(h: int, w: int, rng: random.Random | None = None) -> np.ndarray:
    """Random ellipse / band in lower face region (simulates surgical mask coverage)."""
    r = rng or random
    mask = np.zeros((h, w), dtype=np.float32)
    y0 = int(h * r.uniform(0.22, 0.42))
    y1 = h - int(h * r.uniform(0.0, 0.06))
    cx = w // 2 + int(r.uniform(-w * 0.12, w * 0.12))
    ax = int(w * r.uniform(0.26, 0.44))
    ay = max(8, int((y1 - y0) * r.uniform(0.45, 0.95)))
    cy = (y0 + y1) // 2 + int(r.uniform(-h * 0.02, h * 0.04))
    cv2.ellipse(mask, (cx, cy), (ax, ay), angle=0, startAngle=0, endAngle=360, color=1.0, thickness=-1)
    k = r.choice([0, 0, 31, 41])
    if k > 0:
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)


def _augment_face(
    img: np.ndarray,
    rng: random.Random,
    image_size: int,
) -> np.ndarray:
    """Light geometric + color jitter (helps tiny datasets)."""
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    if rng.random() < 0.45:
        ang = rng.uniform(-14.0, 14.0)
        m = cv2.getRotationMatrix2D((image_size / 2, image_size / 2), ang, 1.0)
        img = cv2.warpAffine(img, m, (image_size, image_size), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.7:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.82, 1.18), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.85, 1.15), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img


class SyntheticMaskFaceDataset(Dataset):
    """Loads face images; applies random synthetic mask; returns (4-ch input, 3-ch target)."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        augment: bool = True,
        extra_exclude: tuple[str, ...] = (),
        auto_exclude_masked: bool = True,
        train_list_path: Path | None = None,
        only_names: tuple[str, ...] | None = None,
    ) -> None:
        self.paths = collect_face_images(
            root,
            extra_exclude,
            auto_exclude_masked,
            train_list_path=train_list_path,
            only_names=only_names,
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No training images under {root} after filters. Add uncovered face JPG/PNG files, "
                "or use --exclude only for the real-mask photo; use --no-auto-exclude if heuristics skip too much."
            )
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.paths[idx % len(self.paths)]
        img = cv2.imread(str(p))
        if img is None:
            return self.__getitem__((idx + 1) % len(self.paths))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        rng = random.Random(hash((str(p), idx)) & 0xFFFFFFFF)
        if self.augment:
            img = _augment_face(img, rng, self.image_size)

        mask = random_surgical_mask(self.image_size, self.image_size, rng=rng)

        rgb = img.astype(np.float32) / 255.0
        m = mask.astype(np.float32) / 255.0
        m = np.clip(m, 0.0, 1.0)
        known = hole_fill_rgb01(rgb, m)

        inp = np.concatenate([known, m[..., None]], axis=-1)
        inp_t = torch.from_numpy(inp).permute(2, 0, 1).float()
        inp_t[:3] = inp_t[:3] * 2.0 - 1.0

        tgt = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tgt = tgt * 2.0 - 1.0

        m_t = torch.from_numpy(m).unsqueeze(0).float()
        return inp_t, tgt, m_t
