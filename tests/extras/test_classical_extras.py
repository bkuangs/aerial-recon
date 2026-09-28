"""Tests for pieces added beyond the milestone gates: P3P, planar PnP, voxel fusion, and the
reference-evaluation helpers."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerial_recon import synthetic
from aerial_recon.eval.reference import icp, vertical_offset
from aerial_recon.geometry.alignment import Sim3
from aerial_recon.geometry.pnp import pnp_p3p, pnp_ransac
from aerial_recon.mvs.fusion import voxel_downsample


def rot_err_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


@pytest.mark.parametrize("planar", [False, True])
def test_p3p_recovers_pose(planar):
    rng = np.random.default_rng(0)
    solved = 0
    for trial in range(50):
        R = Rotation.random(random_state=trial).as_matrix()
        t = rng.normal(size=3) + [0, 0, 10]
        X = rng.normal(size=(3, 3)) * 3
        if planar:
            X[:, 2] = 0.0
        Xc = X @ R.T + t
        if (Xc[:, 2] <= 0).any():
            continue
        poses = pnp_p3p(X, Xc[:, :2] / Xc[:, 2:])
        if any(rot_err_deg(p.R, R) < 1e-4 and np.allclose(p.t, t, atol=1e-5) for p in poses):
            solved += 1
    assert solved >= 45


def test_pnp_ransac_on_planar_scene():
    """DLT resection degenerates on coplanar points; the P3P-based RANSAC must not."""
    rng = np.random.default_rng(3)
    cam = synthetic.default_camera()
    pose = synthetic.look_at(np.array([5.0, -8.0, 40.0]), np.zeros(3))
    X = np.column_stack([rng.uniform(-15, 15, (300, 2)), np.zeros(300)])
    Xc = pose.transform(X)
    uv = Xc[:, :2] / Xc[:, 2:] * cam.fx + [cam.cx, cam.cy] + rng.normal(0, 0.5, (300, 2))
    outlier = rng.random(300) < 0.3
    uv[outlier] = rng.uniform(0, 480, (outlier.sum(), 2))
    est, inliers = pnp_ransac(X, uv, cam, threshold_px=3.0, rng=np.random.default_rng(0))
    assert est is not None
    assert rot_err_deg(est.R, pose.R) < 0.2
    assert np.linalg.norm(est.center - pose.center) < 0.3
    assert np.mean(inliers == ~outlier) > 0.95


def test_voxel_downsample_averages():
    pts = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [1.0, 1.0, 1.0]])
    cols = np.array([[0, 0, 0], [100, 100, 100], [255, 255, 255]])
    out, out_cols = voxel_downsample(pts, cols, 0.1)
    order = np.argsort(out[:, 0])
    np.testing.assert_allclose(out[order], [[0.02, 0.02, 0.02], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(out_cols[order], [[50, 50, 50], [255, 255, 255]])


def test_icp_and_vertical_offset_recover_shift():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-20, 20, (20000, 2))
    z = synthetic.terrain_height(xy, relief=3.0, extent=20.0)
    ref = np.column_stack([xy, z])
    R = Rotation.from_euler("z", 0.5, degrees=True).as_matrix()
    moved = ref @ R.T + [0.4, -0.3, 6.0]
    assert vertical_offset(moved, ref) == pytest.approx(6.0, abs=0.3)
    init = Sim3(1.0, np.eye(3), np.array([0.0, 0.0, -vertical_offset(moved, ref)]))
    T = icp(moved, ref, init=init)
    np.testing.assert_allclose(T.apply(moved), ref, atol=0.05)
