"""M9 gate: depth fusion and geometry metrics. Run: uv run pytest tests/student -k m09"""

import numpy as np
import pytest

from aerial_recon import synthetic
from aerial_recon.eval.geometry import geometry_metrics, nearest_distances
from aerial_recon.mvs.fusion import backproject_depth, consistency_mask, fuse_depth_maps
from aerial_recon.types import Camera

SLABS = [synthetic.Slab(-8, 2, -6, 4, 8.0), synthetic.Slab(4, 12, 2, 10, 4.0)]


def scene():
    cam = Camera.from_fov(160, 120, 60.0)
    centers = [np.array([0.0, 0.0, 40.0]), np.array([6.0, 0.0, 40.0]),
               np.array([-6.0, 0.0, 40.0]), np.array([0.0, 6.0, 40.0])]
    poses = [synthetic.look_at(c, c - [0, 0, 1.0]) for c in centers]
    images, depths = synthetic.render_slab_scene(cam, poses, SLABS)
    return cam, poses, images, depths


def on_surface(points, tol=1e-3):
    z_ok = np.isclose(points[:, 2], 0.0, atol=tol)
    for s in SLABS:
        inside = (points[:, 0] >= s.xmin - tol) & (points[:, 0] <= s.xmax + tol) \
            & (points[:, 1] >= s.ymin - tol) & (points[:, 1] <= s.ymax + tol)
        z_ok |= inside & np.isclose(points[:, 2], s.z, atol=tol)
    return z_ok


def test_backproject_depth_lands_on_surfaces():
    cam, poses, _, depths = scene()
    pts, idx = backproject_depth(depths[0], cam, poses[0])
    assert pts.shape == (np.count_nonzero(depths[0] > 0), 3)
    assert np.all(np.diff(idx) > 0)
    assert on_surface(pts, tol=1e-3).all()


def test_consistency_mask_flags_corrupted_depth():
    cam, poses, _, depths = scene()
    corrupted = depths[0].copy()
    corrupted[40:60, 60:90] *= 1.1
    mask = consistency_mask(corrupted, cam, poses[0], depths[1:], [cam] * 3, poses[1:],
                            min_consistent=2)
    assert mask.dtype == bool and mask.shape == corrupted.shape
    assert not mask[40:60, 60:90].any()
    clean = np.ones_like(mask)
    clean[40:60, 60:90] = False
    assert mask[clean].mean() > 0.8


def test_fused_points_lie_on_surfaces():
    cam, poses, images, depths = scene()
    pts, colors = fuse_depth_maps(depths, [cam] * 4, poses, colors=images, min_consistent=2)
    assert len(pts) > 0.5 * 4 * 160 * 120
    assert colors.shape[0] == len(pts)
    assert on_surface(pts, tol=0.02).mean() > 0.999


def test_nearest_distances():
    ref = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    q = np.array([[1.0, 0, 0], [7.0, 0, 0]])
    np.testing.assert_allclose(nearest_distances(q, ref), [1.0, 3.0])


def test_geometry_metrics():
    gt = np.column_stack([np.linspace(0, 1, 101), np.zeros(101), np.zeros(101)])
    pred = gt[:51] + [0.0, 0.0, 0.005]  # accurate but covers only half the reference
    m = geometry_metrics(pred, gt, threshold=0.01)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(51 / 101)
    assert m["fscore"] == pytest.approx(2 * (51 / 101) / (1 + 51 / 101))
    assert m["accuracy"] == pytest.approx(0.005)
    assert m["completeness"] > 0.1
    assert geometry_metrics(pred, gt, threshold=0.001)["fscore"] == 0.0
