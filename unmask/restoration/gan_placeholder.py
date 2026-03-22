from __future__ import annotations

import numpy as np

from unmask.config import Settings


class GANPlaceholderRestorer:
    """
    Default slot for a GAN / diffusion / custom generative restorer.
    Set UNMASK_GAN_CLASS=module.path:ClassName (see docs/PLUGINS.md).
    """

    name = "gan"

    def restore(self, image_bgr: np.ndarray, mask_255: np.ndarray, settings: Settings) -> np.ndarray:
        raise NotImplementedError(
            "No GAN restorer is bundled. Set UNMASK_GAN_CLASS, or use register() from Python, "
            "or use --backend opencv / unet."
        )
