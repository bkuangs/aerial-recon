"""Image-quality metrics and the shared train/test split. Scaffold.

Both tracks must be evaluated on the *same* held-out views: the MVS mesh is rendered from
the test cameras and compared exactly like the splat renders.
"""

from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray, max_value: float = 1.0,
         mask: np.ndarray | None = None) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff[mask]
    mse = float(np.mean(diff))
    return float("inf") if mse == 0 else 10.0 * np.log10(max_value**2 / mse)


def holdout_split(names: list[str], every: int = 8) -> tuple[list[str], list[str]]:
    """Mip-NeRF 360 convention: sort by name, every `every`-th image is a test view."""
    ordered = sorted(names)
    test = ordered[::every]
    test_set = set(test)
    return [n for n in ordered if n not in test_set], test
