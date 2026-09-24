"""M7 gate: similarity alignment, geodesy, pose metrics. Run: uv run pytest tests/student -k m07"""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerial_recon import synthetic
from aerial_recon.eval.poses import absolute_trajectory_error, pose_auc, relative_pose_errors
from aerial_recon.geo.geodesy import WGS84_A, WGS84_F, ecef_to_enu, lla_to_ecef, lla_to_enu
from aerial_recon.geometry.alignment import Sim3, transform_reconstruction, umeyama
from aerial_recon.types import Pose

RNG = np.random.default_rng(0)


def random_sim3():
    return Sim3(2.5, Rotation.random(random_state=1).as_matrix(), np.array([100.0, -20.0, 5.0]))


def test_umeyama_recovers_similarity():
    T = random_sim3()
    src = RNG.normal(size=(30, 3)) * 10
    est = umeyama(src, T.apply(src))
    assert est.s == pytest.approx(T.s)
    np.testing.assert_allclose(est.R, T.R, atol=1e-9)
    np.testing.assert_allclose(est.t, T.t, atol=1e-8)


def test_umeyama_rigid_and_reflection_safe():
    src = RNG.normal(size=(30, 3))
    R = Rotation.random(random_state=2).as_matrix()
    est = umeyama(src, src @ R.T + 1.0, with_scale=False)
    assert est.s == 1.0
    np.testing.assert_allclose(est.R, R, atol=1e-9)
    mirrored = src * np.array([1.0, 1.0, -1.0])
    assert np.linalg.det(umeyama(src, mirrored).R) == pytest.approx(1.0)


def test_transform_reconstruction_preserves_reprojection():
    points = synthetic.make_terrain_points(seed=0)
    recon = synthetic.make_scene(synthetic.orbit_trajectory(5, 60.0, 40.0), points)
    original = copy.deepcopy(recon)
    T = random_sim3()
    transform_reconstruction(recon, T)
    cam = recon.cameras[1]
    for iid, im in recon.images.items():
        expected_center = T.apply(original.images[iid].pose.center[None])[0]
        np.testing.assert_allclose(im.pose.center, expected_center, atol=1e-8)
        mask = im.point3d_ids >= 0
        X = np.stack([recon.points[p].xyz for p in im.point3d_ids[mask]])
        Xc = im.pose.transform(X)
        uv = Xc[:, :2] / Xc[:, 2:] * cam.fx + [cam.cx, cam.cy]
        np.testing.assert_allclose(uv, im.keypoints[mask], atol=1e-6)


def test_lla_to_ecef_reference_values():
    np.testing.assert_allclose(lla_to_ecef(np.array([[0.0, 0.0, 0.0]])), [[WGS84_A, 0, 0]],
                               atol=1e-6)
    b = WGS84_A * (1 - WGS84_F)
    np.testing.assert_allclose(lla_to_ecef(np.array([[90.0, 0.0, 0.0]])), [[0, 0, b]], atol=1e-6)
    np.testing.assert_allclose(lla_to_ecef(np.array([[0.0, 90.0, 100.0]])),
                               [[0, WGS84_A + 100, 0]], atol=1e-6)


def test_enu_properties():
    ref = np.array([41.31, -81.5, 250.0])
    np.testing.assert_allclose(lla_to_enu(ref[None], ref), [[0, 0, 0]], atol=1e-6)
    up = lla_to_enu(np.array([[41.31, -81.5, 260.0]]), ref)[0]
    np.testing.assert_allclose(up, [0, 0, 10.0], atol=1e-6)
    north = lla_to_enu(np.array([[41.3101, -81.5, 250.0]]), ref)[0]
    assert north[1] == pytest.approx(11.1, abs=0.05) and abs(north[0]) < 1e-6
    east = lla_to_enu(np.array([[41.31, -81.4999, 250.0]]), ref)[0]
    assert east[0] == pytest.approx(8.36, abs=0.05) and abs(east[1]) < 1e-3
    pts = np.column_stack([41.31 + RNG.uniform(-1e-3, 1e-3, 5),
                           -81.5 + RNG.uniform(-1e-3, 1e-3, 5), 250 + RNG.uniform(0, 50, 5)])
    ecef, enu = lla_to_ecef(pts), ecef_to_enu(lla_to_ecef(pts), ref)
    np.testing.assert_allclose(np.linalg.norm(ecef[1:] - ecef[0], axis=1),
                               np.linalg.norm(enu[1:] - enu[0], axis=1), atol=1e-6)


def test_absolute_trajectory_error():
    ref = RNG.normal(size=(40, 3)) * 20
    T = random_sim3()
    noise = RNG.normal(size=ref.shape) * 0.01
    est = (T.apply(ref) + noise * T.s)  # est lives in a scaled, rotated frame
    ate = absolute_trajectory_error(est, ref)
    assert ate.scale == pytest.approx(1 / T.s, rel=1e-3)
    assert ate.rmse == pytest.approx(0.01 * np.sqrt(3), rel=0.25)
    assert ate.errors.shape == (40,)


def test_relative_pose_errors_by_name():
    points = synthetic.make_terrain_points(seed=0)
    ref = synthetic.make_scene(synthetic.orbit_trajectory(5, 60.0, 40.0), points)
    est = copy.deepcopy(ref)
    transform_reconstruction(est, random_sim3())  # errors must be invariant to this
    est.images = {iid + 100: im for iid, im in est.images.items()}  # ids differ, names match
    rot, trans = relative_pose_errors(est, ref)
    assert rot.shape == trans.shape == (10,)
    np.testing.assert_allclose(rot, 0.0, atol=1e-5)
    np.testing.assert_allclose(trans, 0.0, atol=1e-5)
    im = est.images[103]
    im.pose = Pose.from_center(Rotation.from_euler("z", 5, degrees=True).as_matrix() @ im.pose.R,
                               im.pose.center)
    rot, _ = relative_pose_errors(est, ref)
    assert np.sum(np.isclose(rot, 5.0, atol=1e-6)) == 4
    assert np.sum(np.isclose(rot, 0.0, atol=1e-5)) == 6


def test_pose_auc():
    assert pose_auc(np.zeros(10), [5.0, 10.0]) == pytest.approx([1.0, 1.0])
    assert pose_auc(np.array([5.0]), [10.0]) == pytest.approx([0.5])
    assert pose_auc(np.array([1.0, 20.0]), [10.0]) == pytest.approx([0.45])
