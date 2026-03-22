import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from unmask.nn.unet import MaskInpaintUNet


def test_mask_unet_forward() -> None:
    m = MaskInpaintUNet(base=32, legacy_output=False)
    m.eval()
    x = torch.randn(2, 4, 128, 128)
    y = m(x)
    assert y.shape == (2, 3, 128, 128)


def test_mask_unet_legacy_forward() -> None:
    m = MaskInpaintUNet(base=32, legacy_output=True)
    m.eval()
    x = torch.randn(2, 4, 128, 128)
    y = m(x)
    assert y.shape == (2, 3, 128, 128)


if __name__ == "__main__":
    test_mask_unet_forward()
    print("MaskInpaintUNet forward OK")
