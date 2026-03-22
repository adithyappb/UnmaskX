# Contributing

Thank you for taking the time to improve Unmask.

## Getting started

1. Fork the repository and clone it locally.
2. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the test suite:

   ```bash
   python -m pytest tests/
   ```

4. Follow `.gitignore`: do not add face photos, checkpoints, or other large binaries. Use `data/faces/` for local datasets only (see `data/faces/README.md`).

## Pull requests

- Keep changes focused and documented in the PR description.
- Match existing style and typing patterns in the touched files.
- Add or update tests when behavior changes.

## Plugins and custom restorers

Custom inpainting backends integrate through the restoration registry. See [docs/PLUGINS.md](docs/PLUGINS.md). The `gan` slot is reserved for stronger generative models than the default U‑Net.

## Good first issue

GitHub offers an issue template **“Good first issue — Lightweight GAN / generative restorer”** for proposals and implementations that stay within a clear **weight and latency budget** and report **reproducible metrics** (see `utils/metrics.py`).

Reference dataset:

- [Face mask dataset — with and without mask (Kaggle)](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)

## Security

Report security vulnerabilities **privately** per [SECURITY.md](SECURITY.md). Do not file undisclosed issues in public trackers.

## Ethics

Use images only where you have consent and legal right to process them. Do not commit identifiable face data.

