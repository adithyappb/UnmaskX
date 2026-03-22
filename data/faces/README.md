# Local face data

This directory holds **training and evaluation images on your machine**. It follows a simple layout; paths under `data/faces/` are listed in `.gitignore` so large files and personal photos stay out of version control.

## Layout

| Path | Role |
|------|------|
| `train/` | Unmasked faces for U‑Net training (`python -m training.train_unet --data-dir ./data/faces/train`). |
| `test/` | Optional holdout for `eval_unet`. |
| `train_list.txt` | Optional: one filename per line; if omitted or empty, all images in `--data-dir` are used. |

## External datasets

**Kaggle — with/without mask:**  
[Face mask dataset with and without mask](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)

Download and extract locally, then copy **unmasked** crops into `train/` for the default synthetic-mask training pipeline. Paired mask experiments can plug into the [`gan` restorer](../../docs/PLUGINS.md).

## Training excludes

The dataloader skips images under subfolders named `test`, `val`, `holdout`, `exclude`, or `archive_*` when scanning a root, so holdouts and unpacked archives are not mixed into training by accident.

## Setup script

To reorganize screenshots and merge an optional archive (customize filenames in the script if needed):

```bash
python -m training.setup_face_data
```

See the main [README](../../README.md) for full training commands.
