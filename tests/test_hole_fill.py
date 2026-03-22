import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unmask.hole_fill import hole_fill_rgb01


def test_hole_fill_preserves_visible() -> None:
    rgb = np.random.rand(32, 32, 3).astype(np.float32)
    m = np.zeros((32, 32), dtype=np.float32)
    m[10:20, 10:20] = 1.0
    out = hole_fill_rgb01(rgb, m)
    assert np.allclose(out * (1 - m[..., None]), rgb * (1 - m[..., None]))


if __name__ == "__main__":
    test_hole_fill_preserves_visible()
    print("hole_fill OK")
