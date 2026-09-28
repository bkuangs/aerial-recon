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
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, lil_matrix
from scipy.spatial.transform import Rotation

from aerial_recon.geometry.camera import reprojection_errors
from aerial_recon.types import Pose, Reconstruction


@dataclass
class BAReport:
    initial_rmse_px: float
    final_rmse_px: float
    num_observations: int
    num_iterations: int
    success: bool


def reprojection_rmse(recon: Reconstruction) -> float:
    """sqrt(mean |π(X) - u|²) over every observation of every point in registered images."""
    sq_sum, count = 0.0, 0
    by_image: dict[int, tuple[list[int], list[np.ndarray]]] = {}
    for pt in recon.points.values():
        for image_id, kp_idx in pt.track:
            im = recon.images.get(image_id)
            if im is None or not im.is_registered:
                continue
            kps, xyz = by_image.setdefault(image_id, ([], []))
            kps.append(kp_idx)
            xyz.append(pt.xyz)
    for image_id, (kps, xyz) in by_image.items():
        im = recon.images[image_id]
        err = reprojection_errors(recon.cameras[im.camera_id], im.pose, np.asarray(xyz),
                                  im.keypoints[np.asarray(kps)])
        sq_sum += float(np.sum(err**2))
        count += len(err)
    return float(np.sqrt(sq_sum / count)) if count else 0.0


def jacobian_sparsity(
    n_poses: int, n_points: int, pose_index: np.ndarray, point_index: np.ndarray
) -> lil_matrix:
    """Sparsity pattern for least_squares.

    State layout: [pose_0 (6) ... pose_{n-1} (6), point_0 (3) ... point_{m-1} (3)].
    Residual layout: observation k contributes rows 2k (u) and 2k+1 (v), which depend on
    pose block pose_index[k] and point block point_index[k]. Shape: (2K, 6n + 3m).
    Negative indices mark fixed blocks (no columns).
    """
    pose_index = np.asarray(pose_index, dtype=np.int64)
    point_index = np.asarray(point_index, dtype=np.int64)
    n_obs = len(pose_index)
    rows, cols = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
    for r in (0, 1):
        obs = np.flatnonzero(pose_index >= 0)
        for k in range(6):
            rows.append(2 * obs + r)
            cols.append(6 * pose_index[obs] + k)
        obs = np.flatnonzero(point_index >= 0)
        for k in range(3):
            rows.append(2 * obs + r)
            cols.append(6 * n_poses + 3 * point_index[obs] + k)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    shape = (2 * n_obs, 6 * n_poses + 3 * n_points)
    return coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=shape).tolil()


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
    fixed_image_ids = set(fixed_image_ids or ())
    registered = recon.registered_image_ids
    free_ids = [i for i in registered if i not in fixed_image_ids]
    free_index = {iid: k for k, iid in enumerate(free_ids)}
    reg_set = set(registered)

    obs_image, obs_kp, obs_pid = [], [], []
    for pid, pt in recon.points.items():
        for image_id, kp_idx in pt.track:
            if image_id in reg_set:
                obs_image.append(image_id)
                obs_kp.append(kp_idx)
                obs_pid.append(pid)
    initial = reprojection_rmse(recon)
    if not obs_image or (not free_ids and not refine_points):
        return BAReport(initial, initial, len(obs_image), 0, True)

    obs_image = np.asarray(obs_image)
    point_ids = sorted(set(obs_pid)) if refine_points else []
    point_index_of = {pid: k for k, pid in enumerate(point_ids)}
    uv_obs = np.stack([recon.images[i].keypoints[k] for i, k in zip(obs_image, obs_kp,
                                                                     strict=True)])
    pose_index = np.array([free_index.get(int(i), -1) for i in obs_image])
    point_index = (np.array([point_index_of[p] for p in obs_pid]) if refine_points
                   else -np.ones(len(obs_pid), dtype=np.int64))

    # Per-observation intrinsics and the (constant) contribution of fixed parameters.
    cams = [recon.cameras[recon.images[int(i)].camera_id] for i in obs_image]
    fx = np.array([c.fx for c in cams])
    fy = np.array([c.fy for c in cams])
    cx = np.array([c.cx for c in cams])
    cy = np.array([c.cy for c in cams])
    k1 = np.array([c.k1 for c in cams])
    k2 = np.array([c.k2 for c in cams])
    fixed_R = np.stack([recon.images[int(i)].pose.R for i in obs_image])  # noqa: N806
    fixed_t = np.stack([recon.images[int(i)].pose.t for i in obs_image])
    fixed_X = np.stack([recon.points[p].xyz for p in obs_pid])  # noqa: N806
    is_free_pose = pose_index >= 0

    n_poses, n_points = len(free_ids), len(point_ids)
    x0 = np.zeros(6 * n_poses + 3 * n_points)
    for k, iid in enumerate(free_ids):
        pose = recon.images[iid].pose
        x0[6 * k:6 * k + 3] = Rotation.from_matrix(pose.R).as_rotvec()
        x0[6 * k + 3:6 * k + 6] = pose.t
    for k, pid in enumerate(point_ids):
        x0[6 * n_poses + 3 * k:6 * n_poses + 3 * k + 3] = recon.points[pid].xyz

    def residuals(x: np.ndarray) -> np.ndarray:
        poses = x[:6 * n_poses].reshape(-1, 6)
        R = fixed_R.copy()  # noqa: N806
        t = fixed_t.copy()
        if n_poses:
            R_free = Rotation.from_rotvec(poses[:, :3]).as_matrix()  # noqa: N806
            R[is_free_pose] = R_free[pose_index[is_free_pose]]
            t[is_free_pose] = poses[pose_index[is_free_pose], 3:]
        if refine_points:
            X = x[6 * n_poses:].reshape(-1, 3)[point_index]  # noqa: N806
        else:
            X = fixed_X  # noqa: N806
        Xc = np.einsum("nij,nj->ni", R, X) + t  # noqa: N806
        z = Xc[:, 2]
        z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        xy = Xc[:, :2] / z[:, None]
        r2 = np.sum(xy**2, axis=1)
        d = 1.0 + k1 * r2 + k2 * r2**2
        u = fx * d * xy[:, 0] + cx
        v = fy * d * xy[:, 1] + cy
        return np.column_stack([u - uv_obs[:, 0], v - uv_obs[:, 1]]).ravel()

    sparsity = jacobian_sparsity(n_poses, n_points, pose_index, point_index)
    # Anneal the robust scale: start wide so distant starts move, then tighten.
    scales = [loss_scale_px]
    if loss != "linear" and initial > 3.0 * loss_scale_px:
        scales = [3.0 * initial, loss_scale_px]
    x, nfev, success = x0, 0, True
    for f_scale in scales:
        result = least_squares(residuals, x, jac_sparsity=sparsity, method="trf",
                               x_scale="jac", loss=loss, f_scale=f_scale,
                               max_nfev=max_iterations, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                               tr_solver="lsmr")
        x, nfev = result.x, nfev + result.nfev
        success = bool(result.status >= 0)

    poses = x[:6 * n_poses].reshape(-1, 6)
    for k, iid in enumerate(free_ids):
        recon.images[iid].pose = Pose(Rotation.from_rotvec(poses[k, :3]).as_matrix(),
                                      poses[k, 3:])
    if refine_points:
        pts = x[6 * n_poses:].reshape(-1, 3)
        for k, pid in enumerate(point_ids):
            recon.points[pid].xyz = pts[k].copy()
    final_res = residuals(x).reshape(-1, 2)
    per_obs = np.linalg.norm(final_res, axis=1)
    errors: dict[int, list[float]] = {}
    for pid, e in zip(obs_pid, per_obs, strict=True):
        errors.setdefault(pid, []).append(float(e))
    for pid, errs in errors.items():
        recon.points[pid].error = float(np.mean(errs))
    return BAReport(initial, reprojection_rmse(recon), len(obs_pid), nfev, success)
