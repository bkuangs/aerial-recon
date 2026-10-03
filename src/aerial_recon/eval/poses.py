"""M7 — Pose evaluation metrics (used again in M10 and the final study).

Two families:
  * Absolute: align estimated camera centers to reference with Sim3 (umeyama), report
    ATE RMSE in reference units. Sensitive to the alignment; intuitive.
  * Relative: for every image pair, compare relative rotation and translation direction.
    Alignment-free; the standard in learned-pose papers (RRA/RTA, AUC@30° in VGGT).

Reading: Zhang & Scaramuzza (IROS 2018); Jin et al. "Image Matching across Wide Baselines"
(IJCV 2021) for mAA; the VGGT paper (CVPR 2025) §4.1 for AUC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerial_recon.geometry.alignment import umeyama
from aerial_recon.geometry.rotations import rotation_angle_deg
from aerial_recon.types import Reconstruction


@dataclass
class TrajectoryError:
    rmse: float
    median: float
    max: float
    scale: float  # Sim3 scale estimated during alignment
    errors: np.ndarray  # (N,) per-camera center errors after alignment


def absolute_trajectory_error(
    est_centers: np.ndarray, ref_centers: np.ndarray, with_scale: bool = True
) -> TrajectoryError:
    """Align est -> ref with umeyama, then per-camera Euclidean center error."""
    sim = umeyama(est_centers, ref_centers, with_scale=with_scale)
    errors = np.linalg.norm(sim.apply(np.asarray(est_centers)) - ref_centers, axis=1)
    return TrajectoryError(rmse=float(np.sqrt(np.mean(errors**2))),
                           median=float(np.median(errors)), max=float(errors.max()),
                           scale=float(sim.s), errors=errors)


def relative_pose_errors(
    est: Reconstruction, ref: Reconstruction
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation-direction errors (degrees) over all pairs of images
    registered in both reconstructions, matched by image **name**.

    For a pair (i, j) with world-to-camera poses, the relative pose is
    R_ij = R_j R_iᵀ and t_ij = t_j - R_ij t_i. Rotation error is the angle of
    R_ij_estᵀ R_ij_ref; translation error is the angle between t_ij_est and t_ij_ref.
    Returns (rotation_errors (P,), translation_errors (P,)).
    """
    est_by = {im.name: im.pose for im in est.images.values() if im.is_registered}
    ref_by = {im.name: im.pose for im in ref.images.values() if im.is_registered}
    names = sorted(set(est_by) & set(ref_by))

    def rel(pi, pj):
        R = pj.R @ pi.R.T  # noqa: N806
        return R, pj.t - R @ pi.t

    rot, trans = [], []
    for a, name_i in enumerate(names):
        for name_j in names[a + 1:]:
            Re, te = rel(est_by[name_i], est_by[name_j])  # noqa: N806
            Rr, tr = rel(ref_by[name_i], ref_by[name_j])  # noqa: N806
            rot.append(rotation_angle_deg(Re, Rr))
            ne, nr = np.linalg.norm(te), np.linalg.norm(tr)
            if ne < 1e-12 or nr < 1e-12:
                trans.append(180.0)
            else:
                c = np.clip(te @ tr / (ne * nr), -1.0, 1.0)
                trans.append(float(np.degrees(np.arccos(c))))
    return np.asarray(rot), np.asarray(trans)


def pose_auc(errors_deg: np.ndarray, thresholds_deg: list[float]) -> list[float]:
    """Area under the recall-vs-threshold curve, normalized to [0, 1], per threshold τ.

    With the step-function recall curve this is exactly mean_i max(0, 1 - e_i / τ).
    For pairs, pass e = max(rotation_error, translation_error).
    (SuperGlue's reference code integrates with trapezoids and gives slightly different
    numbers; be explicit about which you report.)
    """
    e = np.asarray(errors_deg, dtype=np.float64)
    if e.size == 0:
        return [0.0 for _ in thresholds_deg]
    return [float(np.mean(np.maximum(0.0, 1.0 - e / t))) for t in thresholds_deg]
