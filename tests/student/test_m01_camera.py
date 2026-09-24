"""M1 gate: rotations and camera projection. Run: uv run pytest tests/student -k m01"""

import numpy as np
import pytest

from aerial_recon import synthetic
from aerial_recon.geometry.camera import (
    distort,
    pixel_to_normalized,
    project,
    reprojection_errors,
    undistort,
    unproject,
)
from aerial_recon.geometry.rotations import (
    hat,
    project_to_so3,
    rotation_angle_deg,
    so3_exp,
    so3_log,
)
from aerial_recon.types import Camera

RNG = np.random.default_rng(0)


def random_axis_angle(n, max_angle=np.pi):
    axis = RNG.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return axis * RNG.uniform(0, max_angle, size=(n, 1))


def test_hat_is_cross_product():
    w, v = RNG.normal(size=3), RNG.normal(size=3)
    np.testing.assert_allclose(hat(w) @ v, np.cross(w, v), atol=1e-12)
    np.testing.assert_allclose(hat(w), -hat(w).T)


def test_exp_quarter_turn_about_z():
    R = so3_exp(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_exp_is_a_rotation():
    for w in random_axis_angle(20):
        R = so3_exp(w)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-6, 0.3, 2.0, np.pi - 1e-4, np.pi - 1e-7])
def test_log_inverts_exp(angle):
    axis = np.array([0.3, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    w = axis * angle
    w_back = so3_log(so3_exp(w))
    assert np.all(np.isfinite(w_back))
    np.testing.assert_allclose(so3_exp(w_back), so3_exp(w), atol=1e-8)
    assert np.linalg.norm(w_back) <= np.pi + 1e-9


def test_small_angle_exp_is_first_order_accurate():
    w = np.array([1e-9, -2e-9, 3e-9])
    np.testing.assert_allclose(so3_exp(w), np.eye(3) + hat(w), atol=1e-15)


def test_project_to_so3():
    R = so3_exp(np.array([0.1, 0.2, -0.3]))
    noisy = R + 1e-3 * RNG.normal(size=(3, 3))
    P = project_to_so3(noisy)
    np.testing.assert_allclose(P @ P.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(P) == pytest.approx(1.0)
    assert rotation_angle_deg(P, R) < 0.2
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(project_to_so3(reflection)) == pytest.approx(1.0)


def test_rotation_angle_deg():
    R = so3_exp(np.array([0.0, np.deg2rad(37.0), 0.0]))
    assert rotation_angle_deg(np.eye(3), R) == pytest.approx(37.0)
    assert rotation_angle_deg(R, R) == pytest.approx(0.0, abs=1e-6)


def test_project_matches_synthetic_observations():
    points = synthetic.make_terrain_points(seed=0)
    recon = synthetic.make_scene(synthetic.orbit_trajectory(4, 60.0, 40.0), points)
    cam = recon.cameras[1]
    for im in recon.images.values():
        mask = im.point3d_ids >= 0
        X = np.stack([recon.points[p].xyz for p in im.point3d_ids[mask]])
        uv, depth = project(cam, im.pose, X)
        np.testing.assert_allclose(uv, im.keypoints[mask], atol=1e-9)
        assert np.all(depth > 0)


def test_distortion_value_and_roundtrip():
    cam = Camera(640, 480, 500.0, 500.0, 320.0, 240.0, k1=-0.12, k2=0.03)
    xy = np.array([[0.1, 0.2]])
    r2 = 0.05
    expected = xy * (1 + cam.k1 * r2 + cam.k2 * r2**2)
    np.testing.assert_allclose(distort(cam, xy), expected, atol=1e-15)
    grid = np.stack(np.meshgrid(np.linspace(-0.6, 0.6, 9), np.linspace(-0.45, 0.45, 7)), -1)
    grid = grid.reshape(-1, 2)
    np.testing.assert_allclose(undistort(cam, distort(cam, grid)), grid, atol=1e-9)


def test_project_unproject_roundtrip_with_distortion():
    cam = Camera(640, 480, 500.0, 510.0, 321.0, 239.0, k1=-0.1, k2=0.02)
    pose = synthetic.look_at(np.array([5.0, -20.0, 30.0]), np.zeros(3))
    X = RNG.uniform(-5, 5, size=(100, 3))
    uv, depth = project(cam, pose, X)
    np.testing.assert_allclose(unproject(cam, pose, uv, depth), X, atol=1e-7)
    xn = pixel_to_normalized(cam, uv)
    Xc = pose.transform(X)
    np.testing.assert_allclose(xn, Xc[:, :2] / Xc[:, 2:], atol=1e-9)


def test_reprojection_errors():
    cam = Camera(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = synthetic.look_at(np.array([0.0, -20.0, 30.0]), np.zeros(3))
    X = RNG.uniform(-5, 5, size=(10, 3))
    uv, _ = project(cam, pose, X)
    np.testing.assert_allclose(reprojection_errors(cam, pose, X, uv), 0.0, atol=1e-9)
    np.testing.assert_allclose(reprojection_errors(cam, pose, X, uv + [3.0, 4.0]), 5.0)
