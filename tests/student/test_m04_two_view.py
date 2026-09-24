"""M4 gate: two-view geometry and triangulation. Run: uv run pytest tests/student -k m04"""

import numpy as np
import pytest

from aerial_recon import synthetic
from aerial_recon.geometry.epipolar import (
    decompose_essential,
    essential_from_fundamental,
    estimate_relative_pose,
    fundamental_eight_point,
    sampson_distance,
    select_pose_by_cheirality,
)
from aerial_recon.geometry.triangulation import (
    triangulate_dlt,
    triangulate_multiview,
    triangulation_angles_deg,
)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def angle_deg(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def rot_err_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def two_view(noise=0.0, outlier_ratio=0.0, seed=0, a=1, b=3):
    points = synthetic.make_terrain_points(seed=seed)
    poses = synthetic.orbit_trajectory(12, radius=60.0, altitude=40.0)
    recon = synthetic.make_scene(poses, points, pixel_noise=noise, seed=seed)
    m = synthetic.pairwise_matches(recon, [(a, b)], outlier_ratio=outlier_ratio, seed=seed)[(a, b)]
    gt = {tuple(p) for p in synthetic.correspondences(recon, a, b)}
    is_inlier = np.array([tuple(p) in gt for p in m])
    x1 = recon.images[a].keypoints[m[:, 0]]
    x2 = recon.images[b].keypoints[m[:, 1]]
    p1, p2 = recon.images[a].pose, recon.images[b].pose
    R = p2.R @ p1.R.T
    t = p2.t - R @ p1.t
    return recon, x1, x2, is_inlier, R, t


def test_eight_point_noiseless():
    recon, x1, x2, _, R, t = two_view()
    K = recon.cameras[1].K
    F = fundamental_eight_point(x1, x2)
    assert np.linalg.matrix_rank(F, tol=1e-9 * np.abs(F).max()) == 2
    assert np.linalg.norm(F) == pytest.approx(1.0)
    F_true = np.linalg.inv(K).T @ skew(t) @ R @ np.linalg.inv(K)
    F_true /= np.linalg.norm(F_true)
    assert min(np.linalg.norm(F - F_true), np.linalg.norm(F + F_true)) < 1e-6
    np.testing.assert_allclose(sampson_distance(F, x1, x2), 0.0, atol=1e-6)


def test_sampson_distance_scales_with_noise():
    _, x1, x2, _, _, _ = two_view()
    F = fundamental_eight_point(x1, x2)
    rng = np.random.default_rng(0)
    d = sampson_distance(F, x1 + rng.normal(0, 1.0, x1.shape), x2 + rng.normal(0, 1.0, x2.shape))
    assert 0.3 < np.median(d) < 1.5


def test_essential_and_decomposition():
    recon, x1, x2, _, R, t = two_view()
    K = recon.cameras[1].K
    E = essential_from_fundamental(fundamental_eight_point(x1, x2), K, K)
    s = np.linalg.svd(E, compute_uv=False)
    np.testing.assert_allclose(s / s[0], [1.0, 1.0, 0.0], atol=1e-9)
    candidates = decompose_essential(E)
    assert len(candidates) == 4
    for Rc, tc in candidates:
        assert np.linalg.det(Rc) == pytest.approx(1.0)
        assert np.linalg.norm(tc) == pytest.approx(1.0)
    t_unit = t / np.linalg.norm(t)
    assert any(rot_err_deg(Rc, R) < 1e-4 and angle_deg(tc, t_unit) < 1e-4
               for Rc, tc in candidates)
    xn1 = (np.column_stack([x1, np.ones(len(x1))]) @ np.linalg.inv(K).T)[:, :2]
    xn2 = (np.column_stack([x2, np.ones(len(x2))]) @ np.linalg.inv(K).T)[:, :2]
    Rs, ts, front = select_pose_by_cheirality(candidates, xn1, xn2)
    assert rot_err_deg(Rs, R) < 1e-4 and angle_deg(ts, t_unit) < 1e-4
    assert front.mean() > 0.99


def test_robust_relative_pose():
    recon, x1, x2, is_inlier, R, t = two_view(noise=0.5, outlier_ratio=0.3)
    K = recon.cameras[1].K
    rel = estimate_relative_pose(x1, x2, K, K, threshold_px=1.5, rng=np.random.default_rng(0))
    assert rot_err_deg(rel.R, R) < 0.5
    assert angle_deg(rel.t, t) < 2.0
    precision = np.mean(is_inlier[rel.inliers])
    recall = np.mean(rel.inliers[is_inlier])
    assert precision > 0.95 and recall > 0.85


def test_triangulate_dlt_noiseless():
    recon, _, _, _, _, _ = two_view()
    K = recon.cameras[1].K
    P1 = recon.images[1].pose.projection_matrix(K)
    P2 = recon.images[3].pose.projection_matrix(K)
    m = synthetic.correspondences(recon, 1, 3)
    x1, x2 = recon.images[1].keypoints[m[:, 0]], recon.images[3].keypoints[m[:, 1]]
    X = triangulate_dlt(P1, P2, x1, x2)
    X_true = np.stack([recon.points[p].xyz for p in recon.images[1].point3d_ids[m[:, 0]]])
    np.testing.assert_allclose(X, X_true, atol=1e-6)


def test_triangulate_multiview():
    points = synthetic.make_terrain_points(seed=5)
    recon = synthetic.make_scene(synthetic.orbit_trajectory(8, 60.0, 40.0), points)
    K = recon.cameras[1].K
    pid, pt = max(recon.points.items(), key=lambda kv: len(kv[1].track))
    Ps = [recon.images[i].pose.projection_matrix(K) for i, _ in pt.track]
    xs = np.stack([recon.images[i].keypoints[k] for i, k in pt.track])
    assert len(Ps) >= 3
    np.testing.assert_allclose(triangulate_multiview(Ps, xs), pt.xyz, atol=1e-6)


def test_triangulation_angle():
    ang = triangulation_angles_deg(np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]),
                                   np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1000.0]]))
    assert ang[0] == pytest.approx(90.0)
    assert ang[1] == pytest.approx(np.degrees(2 * np.arctan(1 / 1000)))
