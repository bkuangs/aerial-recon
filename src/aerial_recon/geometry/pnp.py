"""M6 — Perspective-n-Point: registering a new image against the existing map.

Incremental SfM grows the model one image at a time: find 2D-3D matches between the new
image's keypoints and already triangulated points, then solve for the camera pose.

Reading: HZ §7.1–7.2 (DLT camera resection); Lepetit et al. "EPnP" (IJCV 2009);
Kneip et al. "P3P" (CVPR 2011). OpenCV's solvePnPRansac is a fair reference to compare to.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aerial_recon.geometry.alignment import umeyama
from aerial_recon.geometry.camera import pixel_to_normalized, reprojection_errors
from aerial_recon.geometry.ransac import ransac
from aerial_recon.geometry.rotations import project_to_so3
from aerial_recon.types import Camera, Pose


def pnp_dlt(points_world: np.ndarray, xn: np.ndarray) -> Pose | None:
    """Calibrated DLT resection from N >= 6 correspondences.

    xn are *normalized* image coordinates (N, 2) (see camera.pixel_to_normalized). Solve the
    homogeneous 12-unknown system for P = [R | t] up to scale, then fix scale and sign so
    that R is a proper rotation (project_to_so3) and points have positive depth.
    Normalize the 3D points (centroid/scale) for conditioning. Returns None if degenerate.
    """
    X = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)  # noqa: N806
    x = np.asarray(xn, dtype=np.float64).reshape(-1, 2)
    n = len(X)
    if n < 6:
        return None
    c = X.mean(axis=0)
    spread = np.mean(np.linalg.norm(X - c, axis=1))
    if not np.isfinite(spread) or spread < 1e-12:
        return None
    s = np.sqrt(3.0) / spread
    T = np.eye(4)  # noqa: N806
    T[:3, :3] *= s
    T[:3, 3] = -s * c
    Xh = np.column_stack([s * (X - c), np.ones(n)])  # noqa: N806

    A = np.zeros((2 * n, 12))  # noqa: N806
    A[0::2, 0:4] = -Xh
    A[0::2, 8:12] = x[:, 0:1] * Xh
    A[1::2, 4:8] = -Xh
    A[1::2, 8:12] = x[:, 1:2] * Xh
    _, sv, Vt = np.linalg.svd(A)  # noqa: N806
    if sv[-2] < 1e-12 * sv[0]:  # null space > 1-D (e.g. too few / degenerate points)
        return None
    P = Vt[-1].reshape(3, 4) @ T  # noqa: N806

    M = P[:, :3]  # noqa: N806
    if np.linalg.det(M) < 0:
        P = -P  # noqa: N806
        M = -M  # noqa: N806
    scale = np.mean(np.linalg.svd(M, compute_uv=False))
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    R = project_to_so3(M / scale)  # noqa: N806
    t = P[:, 3] / scale
    depth = X @ R[2] + t[2]
    if np.mean(depth > 0) < 0.5:
        return None
    return Pose(R, t)


def pnp_p3p(points_world: np.ndarray, xn: np.ndarray) -> list[Pose]:
    """Minimal calibrated resection from exactly 3 points (Grunert 1841, as reviewed in
    Haralick et al. "Review and analysis of solutions of the three point perspective pose
    estimation problem", IJCV 1994). Returns up to 4 candidate poses.

    Unlike the 12-unknown DLT, P3P does not care whether the points are coplanar, which
    matters here: a beach or a nadir survey gives near-planar structure, and DLT PnP
    degenerates exactly like the 8-point F does in two-view geometry.

    With a = |P2-P3|, b = |P1-P3|, c = |P1-P2| and the ray angles α (j2, j3), β (j1, j3),
    γ (j1, j2), the distances s2 = u s1, s3 = v s1 satisfy a quartic in v. Each positive
    root gives camera-frame points s_i j_i, and absolute orientation (Umeyama, no scale)
    recovers (R, t).
    """
    P = np.asarray(points_world, dtype=np.float64).reshape(3, 3)  # noqa: N806
    j = np.column_stack([np.asarray(xn, dtype=np.float64).reshape(3, 2), np.ones(3)])
    j /= np.linalg.norm(j, axis=1, keepdims=True)
    a2 = np.sum((P[1] - P[2]) ** 2)
    b2 = np.sum((P[0] - P[2]) ** 2)
    c2 = np.sum((P[0] - P[1]) ** 2)
    if min(a2, b2, c2) < 1e-18:
        return []
    ca, cb, cg = j[1] @ j[2], j[0] @ j[2], j[0] @ j[1]
    amc, apc = (a2 - c2) / b2, (a2 + c2) / b2
    bmc, bma = (b2 - c2) / b2, (b2 - a2) / b2
    coeffs = [
        (amc - 1) ** 2 - 4 * c2 / b2 * ca**2,
        4 * (amc * (1 - amc) * cb - (1 - apc) * ca * cg + 2 * c2 / b2 * ca**2 * cb),
        2 * (amc**2 - 1 + 2 * amc**2 * cb**2 + 2 * bmc * ca**2 - 4 * apc * ca * cb * cg
             + 2 * bma * cg**2),
        4 * (-amc * (1 + amc) * cb + 2 * a2 / b2 * cg**2 * cb - (1 - apc) * ca * cg),
        (1 + amc) ** 2 - 4 * a2 / b2 * cg**2,
    ]
    if not np.all(np.isfinite(coeffs)):
        return []
    poses = []
    for v in np.roots(coeffs):
        if abs(v.imag) > 1e-8 * max(1.0, abs(v.real)) or v.real <= 0:
            continue
        v = v.real
        den = 2 * (cg - v * ca)
        if abs(den) < 1e-12:
            continue
        u = ((-1 + amc) * v**2 - 2 * amc * cb * v + 1 + amc) / den
        d = 1 + u * u - 2 * u * cg
        if u <= 0 or d <= 0:
            continue
        s1 = np.sqrt(c2 / d)
        Q = np.stack([s1 * j[0], u * s1 * j[1], v * s1 * j[2]])  # noqa: N806
        T = umeyama(P, Q, with_scale=False)  # noqa: N806
        poses.append(Pose(T.R, T.t))
    return poses


def pnp_ransac(
    points_world: np.ndarray,
    uv: np.ndarray,
    camera: Camera,
    threshold_px: float = 4.0,
    confidence: float = 0.999,
    rng: np.random.Generator | None = None,
    method: str = "p3p",
) -> tuple[Pose | None, np.ndarray]:
    """RANSAC (M3) around a minimal solver with pixel reprojection error.

    method="p3p" (default): sample 4 points, solve P3P on 3 and keep the candidate that
    best reprojects the 4th. Robust to (near-)planar structure. method="dlt": the
    original sample-size-6 DLT, which fails on flat scenes such as Brighton Beach.

    Returns (pose, inliers). The returned pose is already refined by minimizing
    reprojection error over the inliers (a one-image motion-only BA), then inliers are
    recomputed at `threshold_px`.
    """
    X = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)  # noqa: N806
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xn = pixel_to_normalized(camera, uv)
    if method not in ("p3p", "dlt"):
        raise ValueError(f"unknown PnP method {method}")
    sample_size = 4 if method == "p3p" else 6

    def fit(idx: np.ndarray) -> Pose | None:
        if method == "p3p" and len(idx) == 4:
            cands = pnp_p3p(X[idx[:3]], xn[idx[:3]])
            best, best_err = None, np.inf
            for pose in cands:
                xc = pose.transform(X[idx[3:4]])[0]
                if xc[2] <= 0:
                    continue
                err = np.linalg.norm(xc[:2] / xc[2] - xn[idx[3]])
                if err < best_err:
                    best, best_err = pose, err
            return best
        # Refit on all inliers: DLT (may be None on planar data; RANSAC then keeps the
        # minimal-sample model and the nonlinear refinement below does the rest).
        return pnp_dlt(X[idx], xn[idx])

    def residuals(pose: Pose) -> np.ndarray:
        err = reprojection_errors(camera, pose, X, uv)
        depth = X @ pose.R[2] + pose.t[2]
        return np.where(depth > 0, err, np.inf)

    result = ransac(len(X), sample_size, fit, residuals, threshold_px, confidence=confidence,
                    max_iterations=2000, rng=rng)
    if result.model is None or result.inliers.sum() < sample_size:
        return None, np.zeros(len(X), dtype=bool)

    pose, inliers = result.model, result.inliers
    for _ in range(2):
        pose = refine_pose(camera, pose, X[inliers], uv[inliers], threshold_px)
        inliers = residuals(pose) < threshold_px
        if inliers.sum() < 6:
            return None, np.zeros(len(X), dtype=bool)
    return pose, inliers


def refine_pose(
    camera: Camera, pose: Pose, points_world: np.ndarray, uv: np.ndarray, scale_px: float = 4.0
) -> Pose:
    """Nonlinear least-squares pose refinement (6 DoF, robust soft-L1 loss)."""
    from aerial_recon.geometry.camera import project

    def residuals(p: np.ndarray) -> np.ndarray:
        cand = Pose(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:])
        proj, _ = project(camera, cand, points_world)
        return (proj - uv).ravel()

    x0 = np.concatenate([Rotation.from_matrix(pose.R).as_rotvec(), pose.t])
    res = least_squares(residuals, x0, method="trf", loss="soft_l1", f_scale=scale_px / 2,
                        x_scale="jac", max_nfev=50)
    return Pose(Rotation.from_rotvec(res.x[:3]).as_matrix(), res.x[3:])
