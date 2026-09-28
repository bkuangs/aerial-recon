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


def test_voxel_accumulator_buffered_merge_matches_one_shot():
    from aerial_recon.mvs.fusion import _VoxelAccumulator

    rng = np.random.default_rng(1)
    chunks = [(rng.normal(size=(5000, 3)) * 2 - 7, rng.uniform(0, 255, (5000, 3)))
              for _ in range(9)]
    acc = _VoxelAccumulator(0.2, flush_points=7000)  # forces several store merges
    for p, c in chunks:
        acc.add(p, c)
    got, got_cols = acc.result()
    ref, ref_cols = voxel_downsample(np.concatenate([p for p, _ in chunks]),
                                     np.concatenate([c for _, c in chunks]), 0.2)

    def by_voxel(points, cols):
        keys = np.floor(points / 0.2).astype(np.int64)
        order = np.lexsort(keys.T[::-1])
        return keys[order], points[order], cols[order]

    k1, p1, c1 = by_voxel(got, got_cols)
    k2, p2, c2 = by_voxel(ref, ref_cols)
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_allclose(p1, p2, atol=1e-5)
    np.testing.assert_allclose(c1, c2, atol=1e-3)


def test_level_reconstruction_makes_orbit_z_up():
    from aerial_recon.geo.level import level_reconstruction
    from aerial_recon.geometry.alignment import transform_reconstruction
    from aerial_recon.types import Image, Point3D, Pose, Reconstruction

    rng = np.random.default_rng(2)
    recon = Reconstruction()
    for i, az in enumerate(np.linspace(0, 2 * np.pi, 24, endpoint=False)):
        center = np.array([30 * np.cos(az), 30 * np.sin(az), 20.0])
        f = -center / np.linalg.norm(center)  # look at the origin, zero roll
        x = np.cross(f, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(f, x)
        recon.images[i + 1] = Image(i + 1, 1, f"{i}.jpg",
                                    pose=Pose.from_center(np.stack([x, y, f]), center))
    for j, xyz in enumerate(rng.uniform(-10, 10, (200, 3)) * [1, 1, 0.1]):
        recon.points[j + 1] = Point3D(xyz)
    tilt = Rotation.from_rotvec([0.4, -0.7, 1.1]).as_matrix()
    transform_reconstruction(recon, Sim3(0.37, tilt, np.array([5.0, -2.0, 1.0])))

    summary = level_reconstruction(recon)
    centers = np.stack([im.pose.center for im in recon.images.values()])
    assert np.std(centers[:, 2]) < 1e-6  # orbit plane is horizontal again
    assert summary["median_camera_height"] == pytest.approx(0.37 * 20, abs=0.1)
    assert summary["max_abs_roll_deg"] < 1e-6


def test_extract_keyframes_from_panning_video(tmp_path):
    import cv2

    from aerial_recon.video.frames import extract_keyframes

    rng = np.random.default_rng(3)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (240, 1200)).astype(np.uint8), (0, 0), 2)
    path = tmp_path / "pan.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    if not writer.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for i in range(120):  # 4 px / frame pan, 476 px total
        frame = texture[:, 4 * i:4 * i + 320]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    paths, stats = extract_keyframes(path, tmp_path / "kf", min_displacement=40,
                                     analysis_width=320, resize_width=None, search_window=3,
                                     log=lambda _: None)
    assert stats["frames"] == 120
    assert stats["total_displacement_px"] == pytest.approx(476, rel=0.05)
    assert 10 <= len(paths) <= 12
    assert all(p.exists() for p in paths)
    gaps = np.diff(stats["keyframes"])
    assert gaps.min() >= 9 and gaps.max() <= 13


def test_sparse_dense_agreement_in_units():
    from aerial_recon.eval.reference import sparse_dense_agreement

    g = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(g, g)
    dense = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    sparse = np.column_stack([np.full(4, 5.0), np.full(4, 5.0), [0.0, 0.1, 0.3, 0.6]])
    far = np.array([[50.0, 50.0, 0.0]])  # outside the dense footprint: ignored
    out = sparse_dense_agreement(np.vstack([sparse, far]), dense, unit=0.1,
                                 multiples=(2.0, 4.0))
    assert out["num_sparse_in_footprint"] == 4
    assert out["median"] == pytest.approx(2.0)
    assert out["within_2"] == pytest.approx(0.5)
    assert out["within_4"] == pytest.approx(0.75)
