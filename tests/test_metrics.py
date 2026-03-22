import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import composite_quality_index


def test_composite_without_lpips() -> None:
    q = composite_quality_index(25.0, 0.8, None)
    assert 0.0 <= q <= 100.0


def test_composite_with_lpips() -> None:
    q = composite_quality_index(25.0, 0.8, 0.2)
    assert 0.0 <= q <= 100.0


if __name__ == "__main__":
    test_composite_without_lpips()
    test_composite_with_lpips()
    print("composite_quality_index OK")
