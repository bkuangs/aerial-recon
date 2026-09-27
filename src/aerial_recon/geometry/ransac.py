"""M3 — RANSAC: robust estimation when a large fraction of data is garbage.

Every geometric estimator in this project (homography, fundamental matrix, PnP) runs
inside this one loop. Write it once, generically, and test it on a toy model first.

Reading: Fischler & Bolles (1981); HZ §4.7; Raguram et al. "USAC" (TPAMI 2013) for the
modern refinements (LO-RANSAC, PROSAC, MAGSAC) you may add later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RansacResult:
    model: Any
    inliers: np.ndarray  # (N,) bool
    iterations: int


def required_iterations(inlier_ratio: float, sample_size: int, confidence: float) -> int:
    """Number of samples N so that P(at least one all-inlier sample) >= confidence.

    N = log(1 - confidence) / log(1 - inlier_ratio ** sample_size), rounded up.
    Handle the edges: inlier_ratio >= 1 -> 1; inlier_ratio <= 0 -> a very large number
    (return 10**9). Early in RANSAC, inlier_ratio ** sample_size can be ~1e-18, where
    log(1 - p) rounds to 0: use np.log1p(-p) and cap the result at 10**9.
    """
    cap = 10**9
    if inlier_ratio >= 1:
        return 1
    if inlier_ratio <= 0:
        return cap
    denom = np.log1p(-(inlier_ratio**sample_size))
    if denom == 0:
        return cap
    n = np.ceil(np.log1p(-confidence) / denom)
    return int(min(n, cap))


def ransac(
    n_data: int,
    sample_size: int,
    fit: Callable[[np.ndarray], Any | None],
    residuals: Callable[[Any], np.ndarray],
    threshold: float,
    confidence: float = 0.999,
    max_iterations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> RansacResult:
    """Generic RANSAC with adaptive stopping and final least-squares refit.

    Args:
        n_data: number of data points (the loop only deals in indices).
        sample_size: minimal sample size for `fit`.
        fit: indices (k,) -> model or None if degenerate. Must accept k >= sample_size so it
            can be reused to refit on all inliers.
        residuals: model -> (n_data,) non-negative residuals.
        threshold: inlier iff residual < threshold.

    Loop: sample `sample_size` distinct indices, fit, score by inlier count, keep the best,
    and shrink the iteration budget with `required_iterations` using the best inlier ratio.
    Afterwards refit on the best inlier set and recompute inliers (repeat while the inlier
    set grows, at most a few times). If no model is ever found, return model=None and an
    all-False mask.
    """
    rng = rng if rng is not None else np.random.default_rng()
    best_model = None
    best_inliers = np.zeros(n_data, dtype=bool)
    best_count = 0
    if n_data < sample_size:
        return RansacResult(best_model, best_inliers, 0)

    budget = max_iterations
    it = 0
    while it < budget:
        it += 1
        idx = rng.choice(n_data, size=sample_size, replace=False)
        model = fit(idx)
        if model is None:
            continue
        inliers = residuals(model) < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_model, best_inliers, best_count = model, inliers, count
            budget = min(max_iterations,
                         required_iterations(count / n_data, sample_size, confidence))

    if best_model is None:
        return RansacResult(None, np.zeros(n_data, dtype=bool), it)

    for _ in range(5):
        model = fit(np.flatnonzero(best_inliers))
        if model is None:
            break
        inliers = residuals(model) < threshold
        count = int(inliers.sum())
        # Keep the refit only if it doesn't lose support; stop once the set stops growing.
        if count < best_count:
            break
        best_model, best_inliers = model, inliers
        grew = count > best_count
        best_count = count
        if not grew:
            break

    return RansacResult(best_model, best_inliers, it)
