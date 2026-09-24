"""M5 gate: bundle adjustment. Run: uv run pytest tests/student -k m05"""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerial_recon import synthetic
from aerial_recon.sfm.bundle_adjustment import bundle_adjust, jacobian_sparsity, reprojection_rmse
from aerial_recon.types import Pose


def rot_err_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def scene(noise=0.5, seed=0):
    points = synthetic.make_terrain_points(synthetic.TerrainSpec(n_ground_points=500), seed=seed)
    poses = synthetic.orbit_trajectory(10, radius=60.0, altitude=40.0)
    return synthetic.make_scene(poses, points, pixel_noise=noise, seed=seed)


def perturb(recon, rot_deg=1.0, center_m=1.0, point_m=0.5, fixed=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    noisy = copy.deepcopy(recon)
    for iid, im in noisy.images.items():
        if iid in fixed:
            continue
        dR = Rotation.from_rotvec(np.deg2rad(rot_deg) * rng.normal(size=3) / np.sqrt(3))
        R = dR.as_matrix() @ im.pose.R
        im.pose = Pose.from_center(R, im.pose.center + center_m * rng.normal(size=3))
    for pt in noisy.points.values():
        pt.xyz = pt.xyz + point_m * rng.normal(size=3)
    return noisy


def test_jacobian_sparsity_structure():
    pose_idx = np.array([0, 0, 1, 2])
    point_idx = np.array([0, 1, 1, 3])
    A = jacobian_sparsity(3, 4, pose_idx, point_idx)
    assert A.shape == (8, 3 * 6 + 4 * 3)
    assert A.nnz == 4 * 2 * (6 + 3)
    dense = A.toarray() != 0
    assert dense[2, 0:6].all() and not dense[2, 6:18].any()  # obs 1 -> pose 0
    assert dense[5, 18 + 3:18 + 6].all()  # obs 2 -> point 1


def test_reprojection_rmse():
    assert reprojection_rmse(scene(noise=0.0)) == pytest.approx(0.0, abs=1e-9)
    assert reprojection_rmse(scene(noise=0.5)) == pytest.approx(0.5 * np.sqrt(2), rel=0.1)


def test_bundle_adjustment_converges():
    gt = scene(noise=0.5)
    noisy = perturb(gt)
    report = bundle_adjust(noisy, fixed_image_ids={1, 2})
    assert report.initial_rmse_px > 5.0
    assert report.final_rmse_px < 0.75
    assert report.success
    for iid, im in noisy.images.items():
        assert rot_err_deg(im.pose.R, gt.images[iid].pose.R) < 0.05
        assert np.linalg.norm(im.pose.center - gt.images[iid].pose.center) < 0.1
    np.testing.assert_allclose(noisy.images[1].pose.matrix(), gt.images[1].pose.matrix())


def test_motion_only_keeps_points_fixed():
    gt = scene(noise=0.0)
    noisy = perturb(gt, point_m=0.0)
    before = {pid: p.xyz.copy() for pid, p in noisy.points.items()}
    report = bundle_adjust(noisy, fixed_image_ids=set(), refine_points=False)
    assert report.final_rmse_px < 1e-4
    for pid, p in noisy.points.items():
        np.testing.assert_array_equal(p.xyz, before[pid])


def test_robust_loss_survives_outlier_observations():
    gt = scene(noise=0.5, seed=1)
    noisy = perturb(gt, seed=1)
    rng = np.random.default_rng(1)
    for im in noisy.images.values():
        bad = rng.random(len(im.keypoints)) < 0.03
        im.keypoints[bad] += rng.uniform(-40, 40, size=(bad.sum(), 2))
    bundle_adjust(noisy, fixed_image_ids={1, 2}, loss="soft_l1", loss_scale_px=1.0)
    for iid, im in noisy.images.items():
        assert rot_err_deg(im.pose.R, gt.images[iid].pose.R) < 0.2
