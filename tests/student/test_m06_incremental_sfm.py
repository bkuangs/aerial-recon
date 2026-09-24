"""M6 gate: tracks, PnP, and incremental SfM. Run: uv run pytest tests/student -k m06"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from aerial_recon import synthetic
from aerial_recon.geometry.pnp import pnp_dlt, pnp_ransac
from aerial_recon.sfm.incremental import IncrementalSfM
from aerial_recon.sfm.tracks import UnionFind, build_tracks
from aerial_recon.types import Image


def rot_err_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def similarity_align(src, dst):
    """Align src -> dst with scale; kept deliberately opaque (you write the real one in M7)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    s = np.sqrt((b**2).sum() / (a**2).sum())
    rot, _ = Rotation.align_vectors(b, s * a)
    return lambda x: s * rot.apply(x - mu_s) + mu_d


def test_union_find():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("c", "d")
    uf.union("b", "d")
    assert uf.find("a") == uf.find("c")
    uf.union("x", "y")
    assert uf.find("x") != uf.find("a")


def test_build_tracks_chain_and_conflict():
    matches = {
        (1, 2): np.array([[0, 5], [1, 6]]),
        (2, 3): np.array([[5, 9], [6, 7]]),
        (1, 3): np.array([[1, 8]]),  # makes the second track contain image 3 twice
    }
    tracks = build_tracks(matches)
    assert tracks == [[(1, 0), (2, 5), (3, 9)]]


def test_build_tracks_min_length():
    matches = {(1, 2): np.array([[0, 0]]), (3, 4): np.array([[0, 0]]), (4, 5): np.array([[0, 0]])}
    assert build_tracks(matches, min_length=3) == [[(3, 0), (4, 0), (5, 0)]]


def test_build_tracks_on_ground_truth_matches():
    points = synthetic.make_terrain_points(seed=2)
    recon = synthetic.make_scene(synthetic.orbit_trajectory(8, 60.0, 40.0), points)
    tracks = build_tracks(synthetic.pairwise_matches(recon, min_matches=1))
    for track in tracks:
        pids = {recon.images[i].point3d_ids[k] for i, k in track}
        assert len(pids) == 1
    assert len(tracks) == len(recon.points)


def pnp_problem(noise=0.0, outlier_ratio=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cam = synthetic.default_camera()
    pose = synthetic.look_at(np.array([30.0, -40.0, 35.0]), np.zeros(3))
    X = np.column_stack([rng.uniform(-15, 15, (300, 2)), rng.uniform(-2, 8, 300)])
    Xc = pose.transform(X)
    uv = Xc[:, :2] / Xc[:, 2:] * cam.fx + [cam.cx, cam.cy] + rng.normal(0, noise, (300, 2))
    outlier = rng.random(300) < outlier_ratio
    uv[outlier] = rng.uniform(0, 640, (outlier.sum(), 2))
    return cam, pose, X, uv, outlier


def test_pnp_dlt_noiseless():
    cam, pose, X, uv, _ = pnp_problem()
    xn = (uv - [cam.cx, cam.cy]) / cam.fx
    est = pnp_dlt(X[:20], xn[:20])
    assert rot_err_deg(est.R, pose.R) < 1e-4
    np.testing.assert_allclose(est.center, pose.center, atol=1e-6)


def test_pnp_ransac():
    cam, pose, X, uv, outlier = pnp_problem(noise=1.0, outlier_ratio=0.4, seed=1)
    est, inliers = pnp_ransac(X, uv, cam, threshold_px=4.0, rng=np.random.default_rng(0))
    assert rot_err_deg(est.R, pose.R) < 0.3
    assert np.linalg.norm(est.center - pose.center) < 0.5
    assert np.mean(inliers == ~outlier) > 0.95


def run_sfm(poses, points, noise, outlier_ratio, seed=0):
    gt = synthetic.make_scene(poses, points, pixel_noise=noise, seed=seed)
    matches = synthetic.pairwise_matches(gt, outlier_ratio=outlier_ratio, min_matches=30,
                                         seed=seed)
    images = {iid: Image(iid, im.camera_id, im.name, keypoints=im.keypoints.copy())
              for iid, im in gt.images.items()}
    return gt, IncrementalSfM(dict(gt.cameras), images, matches).run()


def check_against_gt(gt, est, max_rot_deg, max_center_frac):
    assert est.registered_image_ids == sorted(gt.images)
    ids = est.registered_image_ids
    align = similarity_align(est.centers(ids), gt.centers(ids))
    extent = np.ptp(gt.centers(ids), axis=0).max()
    center_err = np.linalg.norm(align(est.centers(ids)) - gt.centers(ids), axis=1)
    assert center_err.max() < max_center_frac * extent
    for i in ids:
        for j in ids:
            if i < j:
                Rij_est = est.images[j].pose.R @ est.images[i].pose.R.T
                Rij_gt = gt.images[j].pose.R @ gt.images[i].pose.R.T
                assert rot_err_deg(Rij_est, Rij_gt) < max_rot_deg
    assert len(est.points) > 0.3 * len(gt.points)


@pytest.mark.slow
def test_incremental_sfm_orbit():
    points = synthetic.make_terrain_points(seed=3)
    poses = synthetic.orbit_trajectory(16, radius=60.0, altitude=40.0)
    gt, est = run_sfm(poses, points, noise=0.5, outlier_ratio=0.2)
    check_against_gt(gt, est, max_rot_deg=0.3, max_center_frac=0.005)


@pytest.mark.slow
def test_incremental_sfm_oblique_survey():
    """Survey grid at -60° gimbal over rolling terrain: near-planar, harder to initialize."""
    points = synthetic.make_terrain_points(synthetic.TerrainSpec(n_ground_points=3000), seed=4)
    poses = synthetic.lawnmower_trajectory(rows=3, cols=6, spacing=8.0, altitude=40.0,
                                           gimbal_pitch_deg=-60.0)
    gt, est = run_sfm(poses, points, noise=0.5, outlier_ratio=0.3, seed=4)
    check_against_gt(gt, est, max_rot_deg=0.5, max_center_frac=0.01)
