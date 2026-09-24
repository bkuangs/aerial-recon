"""M8 gate: plane-sweep stereo. Run: uv run pytest tests/student -k m08"""

import numpy as np
import pytest

from aerial_recon import synthetic
from aerial_recon.mvs.plane_sweep import plane_homography, plane_sweep_depth, relative_pose, zncc
from aerial_recon.types import Camera

RNG = np.random.default_rng(0)


def test_relative_pose():
    a = synthetic.look_at(np.array([0.0, -30.0, 40.0]), np.zeros(3))
    b = synthetic.look_at(np.array([10.0, -25.0, 42.0]), np.array([1.0, 0.0, 0.0]))
    R, t = relative_pose(a, b)
    X = RNG.normal(size=(10, 3))
    np.testing.assert_allclose(a.transform(X) @ R.T + t, b.transform(X), atol=1e-9)


def test_plane_homography_maps_plane_points():
    cam = Camera(320, 240, 300.0, 300.0, 160.0, 120.0)
    ref = synthetic.look_at(np.array([0.0, -30.0, 40.0]), np.zeros(3))
    src = synthetic.look_at(np.array([8.0, -28.0, 41.0]), np.zeros(3))
    R, t = relative_pose(ref, src)
    n = np.array([0.1, -0.2, 1.0])
    n /= np.linalg.norm(n)
    d = 45.0
    xy = RNG.uniform(-10, 10, size=(20, 2))
    z = (d - n[0] * xy[:, 0] - n[1] * xy[:, 1]) / n[2]
    X_ref = np.column_stack([xy, z])  # points on the plane, in reference-camera coordinates
    X_src = X_ref @ R.T + t
    u_ref = X_ref[:, :2] / X_ref[:, 2:] * 300.0 + [160.0, 120.0]
    u_src = X_src[:, :2] / X_src[:, 2:] * 300.0 + [160.0, 120.0]
    H = plane_homography(cam.K, cam.K, R, t, n, d)
    mapped = np.column_stack([u_ref, np.ones(20)]) @ H.T
    np.testing.assert_allclose(mapped[:, :2] / mapped[:, 2:], u_src, atol=1e-8)


def test_zncc_properties():
    a = RNG.random((40, 50))
    assert np.allclose(zncc(a, a, 5)[5:-5, 5:-5], 1.0, atol=1e-6)
    assert np.allclose(zncc(a, 3.0 * a + 0.5, 5)[5:-5, 5:-5], 1.0, atol=1e-6)
    assert np.allclose(zncc(a, -a, 5)[5:-5, 5:-5], -1.0, atol=1e-6)
    flat = np.full_like(a, 0.3)
    assert np.allclose(zncc(a, flat, 5), 0.0)
    assert zncc(a, RNG.random((40, 50)), 5).mean() == pytest.approx(0.0, abs=0.05)


def slab_scene():
    cam = Camera.from_fov(160, 120, 60.0)
    centers = [np.array([0.0, 0.0, 40.0]), np.array([5.0, 0.0, 40.0]),
               np.array([-5.0, 0.0, 40.0]), np.array([0.0, 5.0, 40.0])]
    poses = [synthetic.look_at(c, c - [0, 0, 1.0]) for c in centers]
    slabs = [synthetic.Slab(-8, 2, -6, 4, 8.0), synthetic.Slab(4, 12, 2, 10, 4.0)]
    images, depths = synthetic.render_slab_scene(cam, poses, slabs)
    return cam, poses, images, depths


def test_plane_sweep_recovers_slab_depths():
    cam, poses, images, depths = slab_scene()
    hypotheses = np.linspace(28.0, 44.0, 81)
    depth, score = plane_sweep_depth(images[0], cam, poses[0], images[1:], [cam] * 3, poses[1:],
                                     hypotheses, window=7)
    assert depth.shape == score.shape == images[0].shape
    gt = depths[0]
    valid = score > 0.5
    rel = np.abs(depth - gt) / gt
    assert valid.mean() > 0.6
    assert np.median(rel[valid]) < 0.01
    assert np.mean(rel[valid] < 0.03) > 0.9
    roof = valid & (gt < 39.0)  # ground alone would let "depth = 40 everywhere" pass
    assert roof.mean() > 0.1
    assert np.mean(rel[roof] < 0.03) > 0.85
