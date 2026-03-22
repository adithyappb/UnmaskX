"""
One-time layout: move root-level WIN screenshots, merge archive_mask_nomask_dataset into train/,
hold out one image for test/, then remove the archive tree.

Run from repo root:
  python -m training.setup_face_data
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# User's first screenshot — fixed holdout for eval (mask / no-mask pair anchor).
HOLDOUT_NAME = "WIN_20260321_15_02_57_Pro.jpg"


def _force_rmtree(path: Path) -> None:
    """Windows/OneDrive often block shutil.rmtree; try cmd rmdir, then chmod+retry."""
    if not path.is_dir():
        return
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
            check=False,
            capture_output=True,
        )
    if path.is_dir():
        import stat

        for p in path.rglob("*"):
            if p.is_file():
                try:
                    p.chmod(stat.S_IWRITE)
                except OSError:
                    pass
        shutil.rmtree(path, ignore_errors=True)


def _unique_dest(folder: Path, src: Path) -> Path:
    dest = folder / src.name
    if not dest.exists():
        return dest
    h = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:10]
    return folder / f"{src.stem}_{h}{src.suffix}"


def main() -> None:
    faces = ROOT / "data" / "faces"
    archive = faces / "archive_mask_nomask_dataset"
    train = faces / "train"
    test = faces / "test"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)
    marker = faces / ".archive_import_done"

    # 1) Root-level images: holdout -> test/, rest -> train/
    for p in sorted(faces.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMG:
            continue
        if p.name == HOLDOUT_NAME:
            shutil.move(str(p), str(test / p.name))
            print(f"holdout -> test/{p.name}")
        else:
            shutil.move(str(p), str(_unique_dest(train, p)))
            print(f"root -> train/{p.name}")

    # 2) Merge archive into train/ (once; marker avoids duplicate copies if rmdir failed earlier)
    n_copy = 0
    if archive.is_dir() and not marker.is_file():
        for p in archive.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in _IMG:
                continue
            dest = _unique_dest(train, p)
            shutil.copy2(p, dest)
            n_copy += 1
        marker.write_text(f"copied={n_copy}\n", encoding="utf-8")
        print(f"Copied {n_copy} images from archive into train/")
    elif archive.is_dir() and marker.is_file():
        print("Archive copy already done earlier; retrying folder removal only.")

    if archive.is_dir():
        _force_rmtree(archive)
        if archive.is_dir():
            print(f"Warning: could not delete {archive} (close OneDrive sync / Explorer and remove manually).")
        else:
            print(f"Removed {archive.relative_to(ROOT)}")

    # 3) train_list.txt: optional listing — leave a stub so full-folder scan is used
    tlist = faces / "train_list.txt"
    tlist.write_text(
        "# Optional: one filename per line under --data-dir. Empty = scan all images in folder.\n",
        encoding="utf-8",
    )
    print(f"Wrote {tlist.relative_to(ROOT)}")
    print("Done. Train with: python -m training.train_unet --data-dir ./data/faces/train")


if __name__ == "__main__":
    main()
