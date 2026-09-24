"""M3 — Homographies: the geometry of planes (and of nadir drone imagery).

A nadir drone over flat terrain sees an almost planar scene, so image pairs are related by
a homography. That is good for mosaicking and bad for SfM: the fundamental matrix becomes
degenerate. COLMAP scores both models (GRIC-style) to decide how to initialize; you'll
need the homography for the same reason.

Reading: HZ ch. 4 (DLT, normalization, §4.4) and §13.1; Szeliski 2e §8.1.
"""

from __future__ import annotations

import numpy as np


def normalize_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: translate centroid to origin, scale mean distance to sqrt(2).

    Returns (x_normalized (N, 2), T (3, 3)) with x_normalized_h = T @ x_h.
    Shared with the 8-point algorithm in M4.
    """
    raise NotImplementedError("M3: implement normalize_points")


def homography_dlt(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Normalized DLT: (N >= 4) correspondences x2 ~ H x1 -> H (3, 3) with H[2, 2] = 1.

    Return None for degenerate samples (e.g. collinear points -> near-singular H), so that
    RANSAC skips them instead of crashing in transfer_error.
    """
    raise NotImplementedError("M3: implement homography_dlt")


def apply_homography(H: np.ndarray, x: np.ndarray) -> np.ndarray:  # noqa: N803
    """Map points (N, 2) through H with the homogeneous divide."""
    raise NotImplementedError("M3: implement apply_homography")


def transfer_error(H: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:  # noqa: N803
    """Symmetric transfer error: 0.5 * (|x2 - H x1| + |x1 - H⁻¹ x2|) in pixels, shape (N,)."""
    raise NotImplementedError("M3: implement transfer_error")
