"""M3 — Homographies: the geometry of planes (and of nadir drone imagery).

A nadir drone over flat terrain sees an almost planar scene, so image pairs are related by
a homography. That is good for mosaicking and bad for SfM: the fundamental matrix becomes
degenerate. COLMAP scores both models (GRIC-style) to decide how to initialize; you'll
need the homography for the same reason.

Reading: HZ ch. 4 (DLT, normalization, §4.4) and §13.1; Szeliski 2e §8.1.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _has_collinear_triple(x: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check for collinearity between points, which would form a degenerate homography.
    """
    for a, b, c in combinations(range(len(x)), 3):
        u, v = x[b] - x[a], x[c] - x[a]
        if abs(u[0] * v[1] - u[1] * v[0]) < tol:
            return True

    return False


def normalize_points(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Raw pixel coordinates can have very different numerical scales. If the matrix
    A is poorly scaled, then SVD becomes more sensitive to noise and floating-point error.

    We will use Hartley normalization: the centroid of the points becomes the origin, and 
    the points are scaled so their average distance from the origin is roughly sqrt(2).

    Usage: x_norm = T @ x_h

    Returns normalized coordinates and the normalization transform matrix.

    We return T because we need it to undo the normalization afterwards as well.
    """
    c = x.mean(axis=0)
    m = np.mean(np.linalg.norm(x - c, axis=1))
    s = np.sqrt(2) / m if m > 0 else 1.0

    T = np.array([
        [s, 0.0, -s * c[0]],
        [0.0, s, -s * c[1]],
        [0.0, 0.0, 1.0],
    ])
    return s * (x - c), T


def homography_dlt(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Direct linear transform using SVD. Calculate the homography matrix given corresponding
    points from two images.

    Return None for degenerate samples (e.g. collinear points -> near-singular H).
    """
    x1n, T1 = normalize_points(x1)
    x2n, T2 = normalize_points(x2)
    if len(x1) == 4 and (_has_collinear_triple(x1n) or _has_collinear_triple(x2n)): # check collinearity
        return None

    # Build the A matrix
    A = []
    for (x, y), (u, v) in zip(x1n, x2n):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)           # right singular vector of the smallest singular value
    H = np.linalg.inv(T2) @ Hn @ T1     # undo normalization (back to pixel coordinates)

    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12 or np.linalg.cond(H) > 1e12:
        return None

    # Since homographies are only defined up to scale, we want to force the bottom 
    # right entry to be 1 to make H more deterministic
    return H / H[2, 2]


def apply_homography(H: np.ndarray, x: np.ndarray) -> np.ndarray:  # noqa: N803
    """Map points (N, 2) through H with the homogeneous divide."""
    x_h = np.hstack([
        x, 
        np.ones((x.shape[0], 1))
    ])

    x_t = x_h @ H.T
    x_t = x_t[:, :2] / x_t[:, 2:3]   # dehomogenize

    return x_t


def transfer_error(H: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:  # noqa: N803
    """
    Distance, in pixels, that each matched point landed from where H said it should be.

    Symmetric transfer error: 0.5 * (|x2 - H x1| + |x1 - H⁻¹ x2|) in pixels, shape (N,).
    - Forward term H: Map point i from image1 to image2 and see how far it landed
    - Backward term H^-1: Do the same in reverse
    """
    forward = np.linalg.norm(x2 - apply_homography(H, x1), axis=1)
    backward = np.linalg.norm(x1 - apply_homography(np.linalg.inv(H), x2), axis=1)
    return 0.5 * (forward + backward)
