"""M3 gate: matching, RANSAC, homographies. Run: uv run pytest tests/student -k m03"""

import numpy as np
import pytest

from aerial_recon import synthetic
from aerial_recon.geometry.homography import (
    apply_homography,
    homography_dlt,
    normalize_points,
    transfer_error,
)
from aerial_recon.geometry.ransac import ransac, required_iterations
from aerial_recon.sfm.features import detect_sift, match_descriptors, root_sift
from aerial_recon.types import Camera

RNG = np.random.default_rng(0)


def test_root_sift_hellinger_identity():
    a, b = RNG.random((2, 128)).astype(np.float32) * 255
    ra, rb = root_sift(np.stack([a, b]))
    np.testing.assert_allclose(np.linalg.norm(ra), 1.0, atol=1e-6)
    hellinger = np.sum(np.sqrt(a / a.sum() * b / b.sum()))
    assert ra @ rb == pytest.approx(hellinger, rel=1e-5)


def test_match_recovers_permutation_and_rejects_ambiguous():
    desc_a = RNG.random((200, 128)).astype(np.float32)
    perm = RNG.permutation(200)
    desc_b = desc_a[perm] + 0.01 * RNG.normal(size=(200, 128)).astype(np.float32)
    # Make a[0] ambiguous: two near-identical candidates in b.
    j0 = int(np.flatnonzero(perm == 0)[0])
    desc_b = np.vstack([desc_b, desc_b[j0] + 0.001 * RNG.normal(size=128).astype(np.float32)])
    m = match_descriptors(desc_a, desc_b, ratio=0.8, mutual=True)
    assert m.dtype.kind == "i" and m.shape[1] == 2
    found = dict(m.tolist())
    assert 0 not in found
    correct = sum(perm[j] == i for i, j in m.tolist())
    assert correct == len(m) == 199


def test_mutual_check():
    desc_a = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]], dtype=np.float32)
    desc_b = np.array([[0.12, 0.0], [5.0, 5.1], [20.0, 20.0]], dtype=np.float32)
    loose = match_descriptors(desc_a, desc_b, ratio=1.0, mutual=False)
    strict = match_descriptors(desc_a, desc_b, ratio=1.0, mutual=True)
    assert sorted(map(tuple, loose.tolist())) == [(0, 0), (1, 0), (2, 1)]
    assert sorted(map(tuple, strict.tolist())) == [(1, 0), (2, 1)]


def test_required_iterations():
    assert required_iterations(0.5, 4, 0.99) == 72
    assert required_iterations(1.0, 8, 0.999) == 1
    assert required_iterations(0.0, 8, 0.999) >= 10**9
    assert required_iterations(0.3, 8, 0.999) > required_iterations(0.3, 5, 0.999)
    tiny = required_iterations(0.0014, 6, 0.999)  # (0.0014)^6 ~ 7.5e-18
    assert isinstance(tiny, int) and tiny == 10**9


def test_ransac_line_fit_with_half_outliers():
    n = 400
    x = RNG.uniform(-10, 10, n)
    y = 0.5 * x - 2.0 + RNG.normal(0, 0.05, n)
    outlier = RNG.random(n) < 0.5
    y[outlier] = RNG.uniform(-20, 20, outlier.sum())

    def fit(idx):
        A = np.column_stack([x[idx], np.ones(len(idx))])
        return np.linalg.lstsq(A, y[idx], rcond=None)[0]

    def residuals(model):
        return np.abs(model[0] * x + model[1] - y)

    result = ransac(n, 2, fit, residuals, threshold=0.2, rng=np.random.default_rng(1))
    np.testing.assert_allclose(result.model, [0.5, -2.0], atol=0.02)
    assert result.inliers.dtype == bool
    assert np.mean(result.inliers[~outlier]) > 0.98
    assert np.mean(result.inliers[outlier]) < 0.03
    assert result.iterations < 200


def test_normalize_points():
    x = RNG.uniform(0, 4000, size=(50, 2))
    xn, T = normalize_points(x)
    np.testing.assert_allclose(xn.mean(0), 0.0, atol=1e-9)
    assert np.mean(np.linalg.norm(xn, axis=1)) == pytest.approx(np.sqrt(2))
    xh = np.column_stack([x, np.ones(50)]) @ T.T
    np.testing.assert_allclose(xh[:, :2] / xh[:, 2:], xn, atol=1e-9)


def test_homography_dlt_exact():
    H = np.array([[1.1, 0.05, 30.0], [-0.02, 0.95, -12.0], [1e-4, -2e-4, 1.0]])
    x1 = RNG.uniform(0, 640, size=(20, 2))
    x2 = apply_homography(H, x1)
    H_est = homography_dlt(x1, x2)
    np.testing.assert_allclose(H_est, H, atol=1e-8)
    np.testing.assert_allclose(transfer_error(H_est, x1, x2), 0.0, atol=1e-7)


def test_homography_ransac_nadir_flat_ground():
    """Two nadir views of flat ground are exactly related by a homography."""
    cam = synthetic.default_camera()
    poses = [synthetic.look_at(np.array([0.0, 0.0, 50.0]), np.zeros(3)),
             synthetic.look_at(np.array([8.0, 3.0, 52.0]), np.array([8.0, 3.0, 0.0]),
                               up=np.array([0.2, 1.0, 0.0]))]
    xy = RNG.uniform(-25, 25, size=(600, 2))
    ground = np.column_stack([xy, np.zeros(600)])
    recon = synthetic.make_scene(poses, ground, camera=cam, pixel_noise=0.5, seed=0)
    matches = synthetic.pairwise_matches(recon, [(1, 2)], outlier_ratio=0.4, seed=0)[(1, 2)]
    gt = {tuple(p) for p in synthetic.correspondences(recon, 1, 2)}
    is_inlier = np.array([tuple(p) in gt for p in matches])
    x1 = recon.images[1].keypoints[matches[:, 0]]
    x2 = recon.images[2].keypoints[matches[:, 1]]
    res = ransac(len(x1), 4, lambda i: homography_dlt(x1[i], x2[i]),
                 lambda H: transfer_error(H, x1, x2), threshold=3.0,
                 rng=np.random.default_rng(0))
    assert np.mean(res.inliers == is_inlier) > 0.97


@pytest.mark.slow
def test_sift_pipeline_on_rendered_views():
    """End to end: real SIFT on two rendered nadir images, verified by the true homography."""
    cam = Camera.from_fov(480, 360, 60.0)
    poses = [synthetic.look_at(np.array([0.0, 0.0, 30.0]), np.zeros(3)),
             synthetic.look_at(np.array([4.0, 2.0, 30.0]), np.array([4.0, 2.0, 0.0]))]
    images, _ = synthetic.render_slab_scene(cam, poses, [], texels_per_metre=4.0)
    (k1, d1), (k2, d2) = detect_sift(images[0]), detect_sift(images[1])
    m = match_descriptors(root_sift(d1), root_sift(d2))
    assert len(m) > 100
    # Ground-truth homography for the plane z = 0 between the two views.
    H_true = cam.K @ np.array([[1, 0, -4.0 / 30.0], [0, 1, 2.0 / 30.0], [0, 0, 1]]) \
        @ np.linalg.inv(cam.K)
    err = np.linalg.norm(apply_homography(H_true, k1[m[:, 0]]) - k2[m[:, 1]], axis=1)
    assert np.mean(err < 2.0) > 0.9
