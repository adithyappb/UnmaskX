# Restoration plugins

The pipeline selects a **restorer** by id: `opencv`, `unet`, or `gan`.

## Built-in backends

| Id | Description |
|----|-------------|
| `opencv` | OpenCV Navier–Stokes (and optional multiscale blend). No weights. |
| `unet` | `MaskInpaintUNet` with weights at `assets/unmask_unet.pth` by default. |
| `gan` | Extension point: no pretrained GAN is shipped; supply your own class. |

## Implementing a `gan` backend

1. Implement a class with:
   - `restore(self, image_bgr: np.ndarray, mask_255: np.ndarray, settings: unmask.config.Settings) -> np.ndarray`
   - Return a full **BGR `uint8`** image with the masked region filled.
   - Optional: `name = "gan"` for logging.

2. Set the environment variable **`UNMASK_GAN_CLASS`** to `module.path:ClassName`:

   ```bash
   # Windows (cmd)
   set UNMASK_GAN_CLASS=my_package.gan:FaceGANRestorer

   # Unix
   export UNMASK_GAN_CLASS=my_package.gan:FaceGANRestorer

   python app.py --backend gan
   ```

3. Ensure the module is importable (`PYTHONPATH` or an installed package).

## Registering a custom id (Python)

```python
from unmask.restoration.registry import register

class MyRestorer:
    name = "mygan"

    def restore(self, image_bgr, mask_255, settings):
        ...

register("mygan", MyRestorer)
```

To use the CLI, add `"mygan"` to backend choices in your fork, or set `Settings(restoration_backend="mygan")` in code.

## Capabilities and limits

Classical inpainting and small U‑Nets do not guarantee the true appearance of skin under a mask. Heavier priors (GANs, diffusion, hosted APIs) are out of scope for the default weights but fit this plugin model.

**Dataset note:** For paired with/without-mask experiments, see the [Kaggle face-mask dataset](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask) and [`data/faces/README.md`](../data/faces/README.md).
