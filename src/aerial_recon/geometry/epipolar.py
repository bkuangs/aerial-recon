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
from aerial_recon.geometry.homography import normalize_points
from aerial_recon.geometry.triangulation import triangulate_dlt
from aerial_recon.geometry.ransac import ransac

import numpy as np


@dataclass
class RelativePose:
    R: np.ndarray  # (3, 3) rotation taking camera-1 coordinates to camera-2 coordinates
    t: np.ndarray  # (3,) unit-norm translation, x_cam2 = R x_cam1 + t (up to scale)
    inliers: np.ndarray  # (N,) bool, epipolar inliers that also pass cheirality
    F: np.ndarray | None = None


def to_norm(x, K):  # noqa: N803
    return (np.column_stack([x, np.ones(len(x))]) @ np.linalg.inv(K).T)[:, :2]


def fundamental_eight_point(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Use the normalized 8-point algorithm to recover the fundamental matrix. 

    The fundamental matrix for uncalibrated cameras takes a point in one image 
    and maps it to the corresponding EPIPOLAR LINE in the other image; not a point.

    Normalize both point sets (homography.normalize_points), solve Af = 0 by SVD, enforce
    rank 2 by zeroing the smallest singular value, then denormalize: F = T2ᵀ F̂ T1.
    """
    x1n, T1 = normalize_points(x1)
    x2n, T2 = normalize_points(x2)
    u1, v1 = x1n[:, 0:1], x1n[:, 1:2]
    u2, v2 = x2n[:, 0:1], x2n[:, 1:2]

    A = np.column_stack([
        u2 * u1, u2 * v1, u2,
        v2 * u1, v2 * v1, v2,
        u1,      v1,      np.ones_like(u1),
    ])
    _, _, Vt = np.linalg.svd(A)
    Fn = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(Fn)
    S[2] = 0
    Fn = U @ np.diag(S) @ Vt

    F = T2.T @ Fn @ T1
    return F / np.linalg.norm(F)


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:  # noqa: N803
    """
    Sampson distance measures the geometric error of an epipolar correspondence by 
    measuring how far the two measured points need to move so that they satisfy the epipolar constraint.

    d² = (x2ᵀFx1)² / ((Fx1)₁² + (Fx1)₂² + (Fᵀx2)₁² + (Fᵀx2)₂²); return d (not d²), in pixels, shape (N,).
    """
    x1h = np.column_stack([x1, np.ones(len(x1))])
    x2h = np.column_stack([x2, np.ones(len(x2))])

    Fx1 = x1h @ F.T     # row i is F @ x1[i]
    Ftx2 = x2h @ F      # row i is F.T @ x2[i]

    num = np.sum(x2h * Fx1, axis=1)     # x2ᵀ F x1 for each correspondence
    den = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2

    return np.abs(num) / np.sqrt(den)


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:  # noqa: N803
    """E = K2ᵀ F K1, projected onto the essential manifold (singular values (1, 1, 0))."""
    E = K2.T @ F @ K1
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:  # noqa: N803
    """
    The essential matrix is decomposed to recover R and t between two calibrated cameras.

    The decomposition gives four possible camera poses:
    (R_1, +t), (R_1, -t), (R_2, +t), (R_2, -t)

    There are two possible rotations, and translation can point in either direction.

    The essential matrix itself can't tell us which one is correct; we will check which
    of these puts the reconstructed points in front of both cameras (cheirality check).
    """
    U, _, Vt = np.linalg.svd(E)
    # Flipping the sign of U or Vt only negates E (same up to scale) but makes det(R) = +1.
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]     # null vector of Eᵀ, already unit length

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def select_pose_by_cheirality(
    candidates: list[tuple[np.ndarray, np.ndarray]], xn1: np.ndarray, xn2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick the [R t] configuration that puts the most points in front of both cameras.

    xn1/xn2 are normalized image coordinates (N, 2). Returns (R, t, in_front_mask).
    Uses triangulation.triangulate_dlt with P1 = [I | 0], P2 = [R | t].
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    for R, t in candidates:
        P2 = np.hstack([R, t[:, None]])
        X = triangulate_dlt(P1, P2, xn1, xn2)   # in camera-1 coordinates
        z1 = X[:, 2]
        z2 = (X @ R.T + t)[:, 2]
        in_front = np.isfinite(z1) & (z1 > 0) & (z2 > 0)
        if best is None or in_front.sum() > best[2].sum():
            best = (R, t, in_front)

    return best


def estimate_relative_pose(
    x1: np.ndarray,
    x2: np.ndarray,
    K1: np.ndarray,  # noqa: N803
    K2: np.ndarray,  # noqa: N803
    threshold_px: float = 1.0,
    confidence: float = 0.999,
    rng: np.random.Generator | None = None,
) -> RelativePose:
    """
    Recover the relative pose (R, t) between two cameras from noisy pixel matches.

    This ties the whole milestone together:
    1. RANSAC finds F: repeatedly fit fundamental_eight_point to 8 random matches and keep
       the F with the most inliers (matches with sampson_distance < threshold_px).
    2. Convert F to E using the intrinsics K1, K2 (essential_from_fundamental).
    3. Decompose E into its four (R, t) candidates (decompose_essential).
    4. Keep the candidate that puts the inlier points in front of both cameras
       (select_pose_by_cheirality).

    x1, x2 are (N, 2) matched pixel coordinates, already undistorted. t is unit length:
    two images alone can't tell how far apart the cameras are, only in which direction.

    Stretch: compare against cv2.findEssentialMat (5-point + RANSAC) on real image pairs and
    against the homography model on nadir, flat-terrain pairs. When does the 8-point
    estimate break down?
    """
    def fit(idx):
        return fundamental_eight_point(x1[idx], x2[idx])

    def residuals(F):  # noqa: N803
        return sampson_distance(F, x1, x2)

    result = ransac(len(x1), 8, fit, residuals, threshold_px, confidence=confidence, rng=rng)
    F_best, inliers = result.model, result.inliers
    if F_best is None:
        return RelativePose(np.eye(3), np.array([0.0, 0.0, 1.0]), inliers, None)

    E = essential_from_fundamental(F_best, K1, K2)
    idx = np.flatnonzero(inliers)
    xn1, xn2 = to_norm(x1[idx], K1), to_norm(x2[idx], K2)
    R, t, in_front = select_pose_by_cheirality(decompose_essential(E), xn1, xn2)

    # Inliers must satisfy the epipolar constraint *and* cheirality.
    final = np.zeros(len(x1), dtype=bool)
    final[idx[in_front]] = True
    return RelativePose(R, t, final, F_best)
