"""Evaluate a trained MaskInpaintUNet on a folder of face images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.synthetic_dataset import SyntheticMaskFaceDataset
from training.train_unet import try_load_lpips, validate
from unmask.nn.unet import MaskInpaintUNet


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Unmask U-Net checkpoint")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "faces" / "test",
        help="Face image folder (default: holdout test set)",
    )
    p.add_argument("--weights", type=Path, default=ROOT / "assets" / "unmask_unet.pth")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-lpips", action="store_true", help="Skip LPIPS even if installed")
    args = p.parse_args()

    try:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.weights, map_location="cpu")

    base = args.base
    legacy_output = True
    if isinstance(ckpt, dict) and "config" in ckpt:
        base = ckpt["config"].get("base", base)
    if isinstance(ckpt, dict) and "training_config" in ckpt:
        base = ckpt["training_config"].get("base", base)
    if isinstance(ckpt, dict) and ckpt.get("inpaint_forward") == "masked_residual":
        legacy_output = False

    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.no_lpips:
        lpips_fn = None
    else:
        lpips_fn, _ = try_load_lpips(device)

    ds = SyntheticMaskFaceDataset(args.data_dir, image_size=args.image_size, augment=False)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(7))
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MaskInpaintUNet(base=base, legacy_output=legacy_output).to(device)
    model.load_state_dict(state, strict=True)

    vm = validate(model, loader, device, lpips_fn=lpips_fn)
    lp_part = f"  val_lpips={vm.lpips:.4f}" if vm.lpips is not None else ""
    print(
        f"val_l1={vm.l1:.5f}  val_psnr={vm.psnr:.2f}  val_ssim={vm.ssim:.4f}{lp_part}  "
        f"quality_index={vm.quality_index:.2f}/100"
    )
    if isinstance(ckpt, dict) and "quality_index" in ckpt:
        print(f"(checkpoint recorded quality_index={ckpt['quality_index']:.2f})")


if __name__ == "__main__":
    main()
