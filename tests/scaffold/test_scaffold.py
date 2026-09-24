"""Scaffold tests: infrastructure you are given. These pass before you write any code."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aerial_recon import synthetic
from aerial_recon.eval.images import holdout_split, psnr
from aerial_recon.io.colmap_text import read_model, write_model
from aerial_recon.io.exif import read_gps
from aerial_recon.io.ply import read_ply_points, write_ply
from aerial_recon.mvs.warp import warp_image
from aerial_recon.types import Camera, Pose
from aerial_recon.video.frames import extract_frames


def test_pose_center_and_inverse_roundtrip():
    pose = synthetic.look_at(np.array([10.0, -5.0, 30.0]), np.zeros(3))
    np.testing.assert_allclose(pose.center, [10.0, -5.0, 30.0], atol=1e-12)
    np.testing.assert_allclose(pose.compose(pose.inverse()).matrix(), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(pose.R @ pose.R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(pose.R) == pytest.approx(1.0)


def test_look_at_puts_target_on_optical_axis_and_world_up_is_image_up():
    center, target = np.array([0.0, -50.0, 40.0]), np.zeros(3)
    pose = synthetic.look_at(center, target)
    tc = pose.transform(target[None])[0]
    np.testing.assert_allclose(tc[:2], 0.0, atol=1e-9)
    assert tc[2] > 0
    above = pose.transform(np.array([[0.0, 0.0, 5.0]]))[0]
    assert above[1] < 0  # OpenCV: +y is down, so world-up points have negative y


def test_nadir_lawnmower_looks_down():
    poses = synthetic.lawnmower_trajectory(rows=3, cols=4, spacing=10.0, altitude=50.0)
    assert len(poses) == 12
    for pose in poses:
        np.testing.assert_allclose(pose.R[2], [0.0, 0.0, -1.0], atol=1e-9)


def test_synthetic_scene_observations_are_consistent():
    points = synthetic.make_terrain_points(seed=1)
    poses = synthetic.orbit_trajectory(12, radius=60.0, altitude=40.0)
    recon = synthetic.make_scene(poses, points, seed=1)
    cam = recon.cameras[1]
    assert len(recon.points) > 500
    for pid, pt in list(recon.points.items())[:200]:
        assert len(pt.track) >= 2
        for image_id, kp in pt.track:
            im = recon.images[image_id]
            assert im.point3d_ids[kp] == pid
            pc = im.pose.transform(pt.xyz[None])[0]
            uv = cam.K @ pc / pc[2]
            np.testing.assert_allclose(uv[:2], im.keypoints[kp], atol=1e-9)


def test_pairwise_matches_outlier_ratio():
    points = synthetic.make_terrain_points(seed=2)
    recon = synthetic.make_scene(synthetic.orbit_trajectory(6, 60.0, 40.0), points)
    matches = synthetic.pairwise_matches(recon, [(1, 2)], outlier_ratio=0.4, seed=0)
    m = matches[(1, 2)]
    gt = {tuple(p) for p in synthetic.correspondences(recon, 1, 2)}
    inlier_frac = np.mean([tuple(p) in gt for p in m])
    assert inlier_frac == pytest.approx(0.6, abs=0.02)


def test_colmap_text_roundtrip(tmp_path: Path):
    points = synthetic.make_terrain_points(seed=3)
    recon = synthetic.make_scene(synthetic.orbit_trajectory(5, 60.0, 40.0), points)
    recon.cameras[1] = Camera(640, 480, 450.0, 450.0, 320.0, 240.0, k1=-0.05, k2=0.01)
    write_model(recon, tmp_path)
    back = read_model(tmp_path)
    assert back.cameras[1] == recon.cameras[1]
    assert sorted(back.images) == sorted(recon.images)
    for iid, im in recon.images.items():
        np.testing.assert_allclose(back.images[iid].pose.R, im.pose.R, atol=1e-9)
        np.testing.assert_allclose(back.images[iid].pose.t, im.pose.t, atol=1e-7)
        np.testing.assert_array_equal(back.images[iid].point3d_ids, im.point3d_ids)
    assert len(back.points) == len(recon.points)


def test_colmap_text_is_readable_by_pycolmap(tmp_path: Path):
    pycolmap = pytest.importorskip("pycolmap")
    points = synthetic.make_terrain_points(seed=4)
    recon = synthetic.make_scene(synthetic.orbit_trajectory(5, 60.0, 40.0), points)
    write_model(recon, tmp_path)
    rec = pycolmap.Reconstruction(tmp_path)
    assert rec.num_reg_images() == 5
    assert rec.num_points3D() == len(recon.points)
    assert rec.compute_mean_reprojection_error() < 1e-3


def test_ply_roundtrip(tmp_path: Path):
    pts = np.random.default_rng(0).normal(size=(50, 3))
    write_ply(tmp_path / "a.ply", pts, colors=np.full((50, 3), 200, dtype=np.uint8))
    np.testing.assert_allclose(read_ply_points(tmp_path / "a.ply"), pts, atol=1e-6)


def test_exif_gps_roundtrip(tmp_path: Path):
    from PIL import Image as PILImage

    exif = PILImage.Exif()
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (41.0, 18.0, 36.0), 3: "W", 4: (81.0, 30.0, 0.0), 5: 0, 6: 250.5})
    path = tmp_path / "gps.jpg"
    PILImage.new("RGB", (8, 8)).save(path, exif=exif)
    fix = read_gps(path)
    assert fix.latitude_deg == pytest.approx(41.31)
    assert fix.longitude_deg == pytest.approx(-81.5)
    assert fix.altitude_m == pytest.approx(250.5)


def test_extract_frames(tmp_path: Path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    paths = extract_frames(video, tmp_path / "frames", every_n=5)
    assert len(paths) == 4
    assert cv2.imread(str(paths[0])).shape == (48, 64, 3)


def test_rendered_slab_depth_matches_geometry():
    cam = Camera.from_fov(80, 60, 60.0)
    pose = synthetic.look_at(np.array([0.0, 0.0, 30.0]), np.zeros(3))
    slab = synthetic.Slab(-3, 3, -3, 3, 10.0)
    images, depths = synthetic.render_slab_scene(cam, [pose], [slab])
    assert depths[0][30, 40] == pytest.approx(20.0, rel=1e-5)  # roof under image center
    assert depths[0][2, 2] == pytest.approx(30.0, rel=1e-5)  # z-depth of flat ground is constant
    assert 0.0 <= images[0].min() and images[0].max() <= 1.0


def test_warp_identity_and_integer_shift():
    img = np.random.default_rng(0).random((30, 40)).astype(np.float32)
    warped, valid = warp_image(img, np.eye(3), img.shape)
    np.testing.assert_allclose(warped, img, atol=1e-6)
    assert valid.all()
    shift = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    warped, valid = warp_image(img, shift, img.shape)
    np.testing.assert_allclose(warped[:, :30], img[:, 3:33], atol=1e-6)
    assert not valid[:, -3:].any()


def test_psnr_and_holdout_split():
    a = np.zeros((4, 4))
    assert psnr(a, a + 0.1) == pytest.approx(20.0)
    train, test = holdout_split([f"{i:03d}.jpg" for i in range(17)])
    assert test == ["000.jpg", "008.jpg", "016.jpg"]
    assert len(train) == 14


def test_pose_projection_matrix_matches_transform():
    pose = Pose.from_center(np.eye(3), np.array([1.0, 2.0, 3.0]))
    K = np.diag([100.0, 100.0, 1.0])
    X = np.array([[4.0, 5.0, 10.0]])
    x = pose.projection_matrix(K) @ np.append(X[0], 1.0)
    np.testing.assert_allclose(x / x[2], np.append((K @ pose.transform(X)[0])[:2]
                                                   / pose.transform(X)[0, 2], 1.0))
