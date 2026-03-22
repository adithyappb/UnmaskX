<p align="center">
  <strong>Unmask</strong><br/>
  Lower-face inpainting for masked faces — detection, restoration, and training tools.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
</p>

## Overview

Unmask finds the lower-face region occluded by a mask (e.g. surgical-style) using **MediaPipe Face Landmarker**, then fills that region with a pluggable **restorer**: **OpenCV** inpainting, a trained **mask-conditioned U‑Net**, or a **custom plugin** (e.g. GAN) you supply.

Restoration is **approximate**: small models cannot recover a guaranteed true mouth identity. The codebase is built so you can **swap backends** and **measure** runs (PSNR, SSIM, optional LPIPS, composite `quality_index`). See [Contributing](CONTRIBUTING.md) for a labeled entry point toward a lightweight generative restorer.

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.10+**. PyTorch, OpenCV, MediaPipe, and Gradio are listed in `requirements.txt`. The Face Landmarker `.task` file is fetched on first use and is excluded from git (see `.gitignore`).

## Usage

```bash
python app.py                 # Gradio UI
python main.py ui             # same
python main.py webcam         # OpenCV preview
```

| Flag | Behavior |
|------|------------|
| `--backend auto` | U‑Net if `assets/unmask_unet.pth` exists, else OpenCV |
| `--backend unet` / `opencv` | Force a backend |
| `--backend gan` | Load `UNMASK_GAN_CLASS` — [Plugin API](docs/PLUGINS.md) |

The app logs the **effective** backend and checkpoint path on startup.

## Data

**Public dataset (example):** [Face mask dataset — with and without mask (Kaggle)](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)

**Local convention:** place training and holdout images under `data/faces/` as described in [`data/faces/README.md`](data/faces/README.md). That tree is intended for **local** files only; large binaries and photos should not be committed.

Optional layout helper:

```bash
python -m training.setup_face_data
```

## Training

Train the bundled U‑Net on **unmasked** face crops (synthetic masks are applied in the dataloader):

```bash
python -m training.train_unet --epochs 80 --batch-size 4 --base 64 --best-by composite
```

- Default weights path: `assets/unmask_unet.pth` (local artifact; gitignored).
- Metrics: `assets/training_metrics.jsonl`, `assets/last_run_summary.json`.
- `quality_index` (0–100) summarizes masked PSNR/SSIM and optional LPIPS for **run-to-run comparison**, not identity accuracy.

```bash
python -m training.eval_unet --data-dir ./data/faces/test
```

Large datasets on **CPU** with **LPIPS** are slow per epoch; use `tqdm` progress (included in requirements), `--no-lpips`, or GPU for faster iteration.

## Repository layout

| Path | Description |
|------|-------------|
| `unmask/` | Core library: settings, pipeline, detection, `MaskInpaintUNet`, restoration registry |
| `unmask/restoration/` | Built-in `opencv`, `unet`; `gan` plugin hook |
| `training/` | `train_unet`, `setup_face_data`, synthetic dataset |
| `utils/metrics.py` | PSNR, SSIM, `quality_index`, JSONL helpers |
| `assets/` | Local checkpoints and downloaded landmarker (gitignored where heavy) |
| `app.py` | Gradio entrypoint |

Experimental DCGAN scripts under `training/` are not wired into the app.

## What belongs in version control

Binaries and personal images are excluded by design. See `.gitignore` for patterns (e.g. `*.pth`, `assets/*.task`, contents of `data/faces/*` with documented exceptions). Contributors add datasets and weights **locally** or via CI secrets — not in the public tree.

## Contributing

Guidelines, tests, and the **good first issue** template live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Ethics

Process only images you have the right to use. Do not commit identifiable faces or private datasets.

## License

Released under the [MIT License](LICENSE). Update the copyright notice in `LICENSE` when you fork or redistribute.
