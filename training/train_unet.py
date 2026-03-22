"""
Train mask-conditioned U-Net for lower-face inpainting.

Example:
  python -m training.train_unet --data-dir ./data/faces --epochs 80 --batch-size 2

Use **uncovered face** images. Files whose names look like a real masked reference photo
are skipped by default (see training/synthetic_dataset.py). Override with --no-auto-exclude.

Checkpoints include `quality_index` (0–100) and optional `val_lpips` for tracking improvements
across runs; use `--best-by composite` (default) to maximize that score.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.synthetic_dataset import SyntheticMaskFaceDataset
from unmask.nn.unet import MaskInpaintUNet
from utils.metrics import append_metrics_jsonl, composite_quality_index, masked_batch_metrics

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


def try_load_lpips(
    device: torch.device,
) -> tuple[nn.Module | None, Literal["import", "init"] | None]:
    """
    Load LPIPS + AlexNet backbone. On first use torchvision may download AlexNet weights;
    if that fails (firewall, VPN, flaky network → WinError 10054), we return (None, 'init')
    so training can continue with L1-only.
    """
    try:
        import lpips  # type: ignore[import-untyped]
    except ImportError:
        return None, "import"

    try:
        net = lpips.LPIPS(net="alex").to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net, None
    except Exception as e:
        hub_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        print(
            "Note: LPIPS could not load (AlexNet weights download often fails offline or behind strict firewalls).\n"
            f"  {type(e).__name__}: {e}\n"
            "  Training continues with L1-only. Options: use a stable network once, place\n"
            "  alexnet-owt-7be5be79.pth in:\n"
            f"    {hub_ckpt}\n"
            "  from https://download.pytorch.org/models/alexnet-owt-7be5be79.pth\n"
            "  Or pass --no-lpips to skip this message."
        )
        return None, "init"


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred, tgt: N,3,H,W in [-1,1]; mask: N,1,H,W in [0,1]."""
    diff = torch.abs(pred - tgt)
    w = mask.expand_as(pred)
    num = (diff * w).sum()
    den = w.sum() * pred.shape[1] + 1e-6
    return num / den


@dataclass(frozen=True)
class ValMetrics:
    l1: float
    psnr: float
    ssim: float
    lpips: float | None
    quality_index: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    aux_weight: float = 0.06,
    lpips_fn: nn.Module | None = None,
    lpips_weight: float = 0.08,
    desc: str = "train",
) -> float:
    model.train()
    total = 0.0
    n = 0
    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for inp, tgt, m in it:
        inp, tgt, m = inp.to(device), tgt.to(device), m.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(inp)
        loss = masked_l1(pred, tgt, m) + aux_weight * F.l1_loss(pred, tgt)
        if lpips_fn is not None:
            loss = loss + lpips_weight * lpips_fn(pred, tgt).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * inp.size(0)
        n += inp.size(0)
    return total / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lpips_fn: nn.Module | None = None,
    desc: str = "val",
) -> ValMetrics:
    model.eval()
    total_l1 = 0.0
    n = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    if lpips_fn is not None:
        lpips_fn.eval()
    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for inp, tgt, m in it:
        inp, tgt, m = inp.to(device), tgt.to(device), m.to(device)
        pred = model(inp)
        loss = masked_l1(pred, tgt, m)
        total_l1 += float(loss.item()) * inp.size(0)
        n += inp.size(0)
        bpsnr, bssim = masked_batch_metrics(pred, tgt, m)
        psnr_sum += bpsnr * inp.size(0)
        ssim_sum += bssim * inp.size(0)
        if lpips_fn is not None:
            lpips_sum += float(lpips_fn(pred, tgt).mean().item()) * inp.size(0)

    mean_l1 = total_l1 / max(n, 1)
    mean_psnr = psnr_sum / max(n, 1)
    mean_ssim = ssim_sum / max(n, 1)
    mean_lp: float | None = None
    if lpips_fn is not None:
        mean_lp = lpips_sum / max(n, 1)
    qi = composite_quality_index(mean_psnr, mean_ssim, mean_lp)
    return ValMetrics(l1=mean_l1, psnr=mean_psnr, ssim=mean_ssim, lpips=mean_lp, quality_index=qi)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Unmask mask U-Net")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "faces" / "train",
        help="Folder of uncovered face images (recursive; skips test/ val/ subfolders). Default: ./data/faces/train",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--base", type=int, default=48, help="UNet base channels")
    p.add_argument(
        "--val-split",
        type=float,
        default=-1.0,
        help="Fraction for validation (default: 0.1 if >=4 images, else 0)",
    )
    p.add_argument("--out", type=Path, default=ROOT / "assets" / "unmask_unet.pth")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Extra substring; if present in filename, image is excluded (repeatable)",
    )
    p.add_argument(
        "--no-auto-exclude",
        action="store_true",
        help="Do not skip filenames that look like real masked reference shots",
    )
    p.add_argument(
        "--train-list",
        type=Path,
        default=None,
        help="Optional train_list.txt path (default: <data-dir>/train_list.txt if that file exists)",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="If set, only these filenames (or stems) are kept after other filters",
    )
    p.add_argument("--aux-weight", type=float, default=0.06, help="Full-frame L1 weight")
    p.add_argument("--lpips-weight", type=float, default=0.08, help="Perceptual (LPIPS) term weight")
    p.add_argument("--no-lpips", action="store_true", help="Disable LPIPS even if installed")
    p.add_argument(
        "--best-by",
        choices=("composite", "l1", "ssim"),
        default="composite",
        help="Metric used to pick the saved checkpoint (composite = maximize quality_index)",
    )
    p.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=ROOT / "assets" / "training_metrics.jsonl",
        help="Append one JSON object per epoch for long-term tracking",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_status: Literal["import", "init"] | None = None
    if args.no_lpips:
        lpips_fn = None
    else:
        lpips_fn, lpips_status = try_load_lpips(device)
    if lpips_status == "import":
        print("Note: LPIPS not installed (pip install lpips). Training uses L1 only.")

    extra = tuple(str(x) for x in args.exclude if x)
    only = tuple(str(x) for x in args.only if x)
    only_names: tuple[str, ...] | None = only if only else None
    full = SyntheticMaskFaceDataset(
        args.data_dir,
        image_size=args.image_size,
        augment=True,
        extra_exclude=extra,
        auto_exclude_masked=not args.no_auto_exclude,
        train_list_path=args.train_list,
        only_names=only_names,
    )

    n = len(full)
    if n <= 20:
        print("Using training files:", ", ".join(p.name for p in full.paths))
    else:
        preview = ", ".join(p.name for p in full.paths[:5])
        print(f"Using {n} training images (showing first 5): {preview}, ...")

    val_split = args.val_split
    if val_split < 0:
        val_split = 0.1 if n >= 4 else 0.0

    if val_split <= 0:
        train_ds = full
        val_ds = full
        print("Note: val_split=0 (tiny set). Validation metrics are in-sample; still saves best checkpoint.")
    else:
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        if n_train < 1:
            raise SystemExit("Not enough images for train/val split. Use --val-split 0")
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    bs = max(1, min(args.batch_size, len(train_ds)))
    if bs != args.batch_size:
        print(f"Note: batch size capped to {bs} for this dataset.")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(
        f"Each epoch: {len(train_loader)} train batches, {len(val_loader)} val batches | "
        f"device={device} | LPIPS={'on' if lpips_fn else 'off'}"
    )
    if tqdm is None:
        print("Tip: pip install tqdm for a live progress bar (each batch) so long epochs do not look stuck.")
    if device.type == "cpu" and lpips_fn is not None and len(train_loader) > 80:
        print(
            "Note: CPU + LPIPS over many batches is slow; each epoch can take a long time before the next "
            "Epoch N/80 line. Use --no-lpips for faster runs, or train on GPU."
        )

    model = MaskInpaintUNet(base=args.base, legacy_output=False).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_l1 = float("inf")
    best_ssim = -1.0
    best_composite = -1.0
    best_value_for_selection: float | None = None
    run_best_l1 = float("inf")
    run_best_ssim = -1.0
    run_best_qi = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    training_config = {
        "run_id": run_id,
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": bs,
        "lr": args.lr,
        "image_size": args.image_size,
        "base": args.base,
        "val_split": val_split,
        "aux_weight": args.aux_weight,
        "lpips_weight": args.lpips_weight,
        "lpips_enabled": lpips_fn is not None,
        "best_by": args.best_by,
        "out": str(args.out),
    }

    for epoch in range(args.epochs):
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            aux_weight=args.aux_weight,
            lpips_fn=lpips_fn,
            lpips_weight=args.lpips_weight,
            desc=f"Epoch {epoch + 1}/{args.epochs} train",
        )
        vm = validate(
            model,
            val_loader,
            device,
            lpips_fn=lpips_fn,
            desc=f"Epoch {epoch + 1}/{args.epochs} val",
        )
        sched.step()

        lp_str = f"  val_lpips={vm.lpips:.4f}" if vm.lpips is not None else ""
        print(
            f"Epoch {epoch + 1}/{args.epochs}  train_loss={tr:.5f}  val_l1={vm.l1:.5f}  "
            f"val_psnr={vm.psnr:.2f}  val_ssim={vm.ssim:.4f}{lp_str}  "
            f"quality_index={vm.quality_index:.2f}/100"
        )

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "epoch": epoch + 1,
            "train_loss": tr,
            **asdict(vm),
            "best_by": args.best_by,
        }
        append_metrics_jsonl(args.metrics_jsonl, row)

        run_best_l1 = min(run_best_l1, vm.l1)
        run_best_ssim = max(run_best_ssim, vm.ssim)
        run_best_qi = max(run_best_qi, vm.quality_index)

        improved = False
        if args.best_by == "l1":
            if vm.l1 < best_l1:
                best_l1 = vm.l1
                improved = True
                best_value_for_selection = vm.l1
        elif args.best_by == "ssim":
            if vm.ssim > best_ssim:
                best_ssim = vm.ssim
                improved = True
                best_value_for_selection = vm.ssim
        else:
            if vm.quality_index > best_composite:
                best_composite = vm.quality_index
                improved = True
                best_value_for_selection = vm.quality_index

        if improved:
            payload = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_l1": vm.l1,
                "val_psnr": vm.psnr,
                "val_ssim": vm.ssim,
                "val_lpips": vm.lpips,
                "quality_index": vm.quality_index,
                "best_by": args.best_by,
                "best_metric_value": best_value_for_selection,
                "training_config": training_config,
                "config": {"base": args.base},
                "inpaint_forward": "masked_residual",
                "metrics_schema": "unmask_unet_v3",
            }
            torch.save(payload, args.out)
            print(f"  saved best ({args.best_by}) -> {args.out}")

    summary = {
        "run_id": run_id,
        "finished": datetime.now(timezone.utc).isoformat(),
        "best_by": args.best_by,
        "checkpoint_selection_best": best_value_for_selection,
        "epoch_best_val_l1": run_best_l1 if run_best_l1 != float("inf") else None,
        "epoch_best_val_ssim": run_best_ssim if run_best_ssim >= 0 else None,
        "epoch_best_quality_index": run_best_qi if run_best_qi >= 0 else None,
        "weights": str(args.out),
    }
    with (args.out.parent / "last_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Done. Selection={args.best_by!r} best_value={best_value_for_selection!r}. "
        f"Weights: {args.out}  Log: {args.metrics_jsonl}"
    )


if __name__ == "__main__":
    main()
