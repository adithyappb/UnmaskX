---
name: Good first issue — Lightweight GAN / generative restorer
about: Propose or implement an efficient generative backend for lower-face restoration via the plugin API.
title: "[RFC] Lightweight generative restorer for lower-face unmasking"
labels: "good first issue, enhancement, help wanted"
---

## Summary

The default **U‑Net** and **OpenCV** backends produce plausible fills, not guaranteed identity match. This issue tracks work toward a **small, efficient** generative model (or similar) exposed through the **`gan` plugin slot**, with explicit **size and latency** targets and **reported metrics** so others can compare and iterate.

## Proposed scope

- [ ] Implement a restorer matching `restore(image_bgr, mask_255, settings) -> image_bgr` ([`docs/PLUGINS.md`](docs/PLUGINS.md)).
- [ ] State a **parameter / FLOPs / latency** budget in the PR.
- [ ] Document **metrics**: masked PSNR / SSIM, optional LPIPS, and/or the `quality_index` pattern in `utils/metrics.py`; describe the eval split (local holdout under `data/faces/test/` — keep images out of git).
- [ ] Document **weights**: `.gitignore` excludes `*.pth`; provide download instructions or training steps.

## Datasets

- [Kaggle — face mask with and without mask](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)
- Local layout: [`data/faces/README.md`](data/faces/README.md)

## Expectations

Perfect identity recovery under a mask is **not** a requirement for closing this issue; the goal is a **measurable** step forward within a **lightweight** budget.

## References

- [`docs/PLUGINS.md`](docs/PLUGINS.md)
- [`README.md`](README.md)
- [`training/train_unet.py`](training/train_unet.py)
