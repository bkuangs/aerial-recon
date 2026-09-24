"""M5 — Bundle adjustment: the heart of SfM.

Minimize total reprojection error over all camera poses and 3D points:

    min_{R_i, t_i, X_j}  Σ_{(i,j) observed} ρ( | π(K_i, R_i, t_i, X_j) - u_ij |² )

with a robust loss ρ (Huber/Cauchy). Parameterize each rotation by an axis-angle vector
(M1's so3_exp), so the state is 6 numbers per camera + 3 per point. The Jacobian is
extremely sparse: each residual touches exactly one camera block and one point block. That
sparsity (and the Schur complement it enables) is why BA scales to thousands of images.

Here you use scipy.optimize.least_squares(method="trf", jac_sparsity=...). It won't match
Ceres' speed, but it is the same math.

Gauge freedom: the solution is only defined up to a similarity transform (7 DoF). Fixing
one camera removes 6; scale floats (fix a second camera's center or just let the damping
handle it). Think about what happens if you fix nothing.

Reading: Triggs et al. "Bundle Adjustment — A Modern Synthesis" (1999) §1–6 (skim the
rest); Szeliski 2e §11.4; Agarwal et al. "Bundle Adjustment in the Large" (ECCV 2010);
the scipy cookbook "Large-scale bundle adjustment in scipy".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix

from aerial_recon.types import Reconstruction


@dataclass
class BAReport:
    initial_rmse_px: float
    final_rmse_px: float
    num_observations: int
    num_iterations: int
    success: bool


def reprojection_rmse(recon: Reconstruction) -> float:
    """sqrt(mean |π(X) - u|²) over every observation of every point in registered images."""
    raise NotImplementedError("M5: implement reprojection_rmse")


def jacobian_sparsity(
    n_poses: int, n_points: int, pose_index: np.ndarray, point_index: np.ndarray
) -> lil_matrix:
    """Sparsity pattern for least_squares.

    State layout: [pose_0 (6) ... pose_{n-1} (6), point_0 (3) ... point_{m-1} (3)].
    Residual layout: observation k contributes rows 2k (u) and 2k+1 (v), which depend on
    pose block pose_index[k] and point block point_index[k]. Shape: (2K, 6n + 3m).
    """
    raise NotImplementedError("M5: implement jacobian_sparsity")


def bundle_adjust(
    recon: Reconstruction,
    fixed_image_ids: set[int] | None = None,
    refine_points: bool = True,
    loss: str = "soft_l1",
    loss_scale_px: float = 1.0,
    max_iterations: int = 100,
) -> BAReport:
    """Jointly refine registered poses and 3D points **in place**. Intrinsics stay fixed.

    Args:
        fixed_image_ids: poses held constant (at least one for a well-posed gauge).
        refine_points: if False, only poses are optimized (motion-only BA, useful after PnP).
        loss / loss_scale_px: passed to least_squares as `loss` and `f_scale`.

    Only observations in registered images count. Report RMSE before/after using
    reprojection_rmse.

    Tips from building the reference solution:
      * From a far-off start, a robust loss with a 1 px scale barely moves — every
        residual sits in the L1-like regime and the solver stalls. Warm-start with a
        larger scale (e.g. 3x the initial RMSE), then refine at loss_scale_px.
      * scipy's "huber" converged pathologically slowly on these problems (1000
        evaluations, still not done) where "soft_l1" (a smooth Huber) took ~20. Try both
        and explain the difference in your notes (hint: look at ρ'' and how scipy rescales
        residuals and Jacobians for robust losses).
      * Finite-difference Jacobians work but are slow; the analytic 2x6 and 2x3 blocks are
        a worthwhile stretch goal (check them against finite differences).
    """
    raise NotImplementedError("M5: implement bundle_adjust")
