"""M4 — Two-view geometry: from matches to relative pose.

    x2ᵀ F x1 = 0            (pixels)         F = K2⁻ᵀ E K1⁻¹
    x̂2ᵀ E x̂1 = 0            (normalized)     E = [t]x R

E has 5 DoF; recovering (R, t) from it gives 4 candidates, and only one puts the points
in front of both cameras (cheirality). Translation is recovered only up to scale —
the root of monocular scale ambiguity, which GPS will later fix (M7).

Reading: HZ ch. 9 (esp. §9.6), ch. 11 (§11.1–11.4, the normalized 8-point algorithm);
Hartley "In Defense of the Eight-Point Algorithm" (TPAMI 1997);
Nistér "An Efficient Solution to the Five-Point Relative Pose Problem" (TPAMI 2004).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RelativePose:
    R: np.ndarray  # (3, 3) rotation taking camera-1 coordinates to camera-2 coordinates
    t: np.ndarray  # (3,) unit-norm translation, x_cam2 = R x_cam1 + t (up to scale)
    inliers: np.ndarray  # (N,) bool, epipolar inliers that also pass cheirality
    F: np.ndarray | None = None


def fundamental_eight_point(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Normalized 8-point algorithm (N >= 8) -> F (3, 3), rank 2, Frobenius norm 1.

    Normalize both point sets (homography.normalize_points), solve Af = 0 by SVD, enforce
    rank 2 by zeroing the smallest singular value, then denormalize: F = T2ᵀ F̂ T1.
    """
    raise NotImplementedError("M4: implement fundamental_eight_point")


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:  # noqa: N803
    """First-order geometric error of each correspondence w.r.t. F, in pixels, shape (N,).

    d² = (x2ᵀFx1)² / ((Fx1)₁² + (Fx1)₂² + (Fᵀx2)₁² + (Fᵀx2)₂²); return d (not d²).
    """
    raise NotImplementedError("M4: implement sampson_distance")


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:  # noqa: N803
    """E = K2ᵀ F K1, projected onto the essential manifold (singular values (1, 1, 0))."""
    raise NotImplementedError("M4: implement essential_from_fundamental")


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:  # noqa: N803
    """The four (R, t) candidates of E, with det(R) = +1 and |t| = 1 (HZ result 9.19)."""
    raise NotImplementedError("M4: implement decompose_essential")


def select_pose_by_cheirality(
    candidates: list[tuple[np.ndarray, np.ndarray]], xn1: np.ndarray, xn2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the candidate that puts the most points in front of both cameras.

    xn1/xn2 are normalized image coordinates (N, 2). Returns (R, t, in_front_mask).
    Uses triangulation.triangulate_dlt with P1 = [I | 0], P2 = [R | t].
    """
    raise NotImplementedError("M4: implement select_pose_by_cheirality")


def estimate_relative_pose(
    x1: np.ndarray,
    x2: np.ndarray,
    K1: np.ndarray,  # noqa: N803
    K2: np.ndarray,  # noqa: N803
    threshold_px: float = 1.0,
    confidence: float = 0.999,
    rng: np.random.Generator | None = None,
) -> RelativePose:
    """Robust relative pose from undistorted pixel correspondences.

    RANSAC (M3) over fundamental_eight_point with sampson_distance < threshold_px, then
    E from F, decompose, and choose by cheirality on the inliers.

    Stretch: compare against cv2.findEssentialMat (5-point + RANSAC) on real image pairs and
    against the homography model on nadir, flat-terrain pairs. When does the 8-point
    estimate break down?
    """
    raise NotImplementedError("M4: implement estimate_relative_pose")
