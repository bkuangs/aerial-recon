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
    raise NotImplementedError("M7: implement absolute_trajectory_error")


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
    raise NotImplementedError("M7: implement relative_pose_errors")


def pose_auc(errors_deg: np.ndarray, thresholds_deg: list[float]) -> list[float]:
    """Area under the recall-vs-threshold curve, normalized to [0, 1], per threshold τ.

    With the step-function recall curve this is exactly mean_i max(0, 1 - e_i / τ).
    For pairs, pass e = max(rotation_error, translation_error).
    (SuperGlue's reference code integrates with trapezoids and gives slightly different
    numbers; be explicit about which you report.)
    """
    raise NotImplementedError("M7: implement pose_auc")
