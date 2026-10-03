"""M7 — Similarity alignment: putting a reconstruction into a known frame.

Monocular SfM output lives in an arbitrary frame with arbitrary scale. Aligning it to GPS
positions (georeferencing) or to a reference reconstruction (evaluation) is a 7-DoF
similarity problem with a closed-form solution.

Reading: Umeyama "Least-squares estimation of transformation parameters between two point
patterns" (TPAMI 1991); Horn "Closed-form solution of absolute orientation" (1987);
Zhang & Scaramuzza "A Tutorial on Quantitative Trajectory Evaluation" (IROS 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerial_recon.types import Pose, Reconstruction


@dataclass(frozen=True)
class Sim3:
    """x' = s R x + t."""

    s: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.s * points @ self.R.T + self.t


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim3 (or SE3 if with_scale=False) with dst ≈ s R src + t.

    src, dst: (N, 3), N >= 3 non-collinear. Remember the reflection fix (det(R) = +1).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)  # noqa: N806
    S = np.eye(3)  # noqa: N806
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt  # noqa: N806
    if with_scale:
        var_s = np.mean(np.sum(a**2, axis=1))
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


def transform_reconstruction(recon: Reconstruction, sim3: Sim3) -> Reconstruction:
    """Apply a similarity to every point and camera **in place** and return recon.

    Points: X' = s R X + t. Cameras keep their image content, so their centers move like
    points, their orientation rotates by R, and world-to-camera t must be recomputed.
    Derive it — this is the most common georeferencing bug.

    With X' = s R_s X + t_s, a camera x_c = R X + t must still see x_c ∝ R' X' + t'.
    Taking R' = R R_sᵀ and C' = s R_s C + t_s gives t' = -R' C'; camera-frame coordinates
    scale by s, which leaves the projection unchanged.
    """
    for pt in recon.points.values():
        pt.xyz = sim3.apply(pt.xyz[None])[0]
    for im in recon.images.values():
        if im.pose is None:
            continue
        R_new = im.pose.R @ sim3.R.T  # noqa: N806
        center = sim3.apply(im.pose.center[None])[0]
        im.pose = Pose.from_center(R_new, center)
    return recon
