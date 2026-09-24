"""Synthetic drone scenes with exact ground truth.

Everything here is scaffold: it exists so every student milestone can be tested against
known answers before touching real footage. Real data hides bugs; synthetic data exposes them.

World frame: local ENU-like (x east, y north, z up), units are metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from aerial_recon.types import Camera, Image, Pose, Reconstruction


def default_camera(width: int = 640, height: int = 480, hfov_deg: float = 70.0) -> Camera:
    return Camera.from_fov(width, height, hfov_deg)


def look_at(center: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> Pose:
    """World-to-camera pose for an OpenCV camera at `center` looking at `target`.

    Image "up" is aligned with world `up` (default +z); for near-nadir views the world +y
    (north) is used instead so that image top faces north.
    """
    center = np.asarray(center, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - center
    z /= np.linalg.norm(z)
    if up is None:
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Pose.from_center(np.stack([x, y, z]), center)


def orbit_trajectory(
    n_views: int,
    radius: float,
    altitude: float,
    target: np.ndarray | None = None,
    arc_deg: float = 360.0,
) -> list[Pose]:
    """Classic "point of interest" drone orbit: circle at fixed altitude, gimbal on target."""
    target = np.zeros(3) if target is None else np.asarray(target, dtype=np.float64)
    endpoint = arc_deg < 360.0
    angles = np.deg2rad(np.linspace(0.0, arc_deg, n_views, endpoint=endpoint))
    poses = []
    for a in angles:
        c = target + np.array([radius * np.cos(a), radius * np.sin(a), altitude])
        poses.append(look_at(c, target))
    return poses


def lawnmower_trajectory(
    rows: int,
    cols: int,
    spacing: float,
    altitude: float,
    gimbal_pitch_deg: float = -90.0,
) -> list[Pose]:
    """Survey/mapping grid flight (the pattern photogrammetry apps fly).

    gimbal_pitch_deg = -90 is nadir (straight down); -60 is a typical oblique mapping angle
    that tilts the view along the direction of travel.
    """
    poses = []
    x0 = -0.5 * (cols - 1) * spacing
    y0 = -0.5 * (rows - 1) * spacing
    pitch = np.deg2rad(gimbal_pitch_deg)
    for r in range(rows):
        heading = 1.0 if r % 2 == 0 else -1.0
        col_order = range(cols) if heading > 0 else range(cols - 1, -1, -1)
        for c in col_order:
            center = np.array([x0 + c * spacing, y0 + r * spacing, altitude])
            forward = np.array([heading * np.cos(pitch), 0.0, np.sin(pitch)])
            poses.append(look_at(center, center + forward))
    return poses


@dataclass
class TerrainSpec:
    extent: float = 40.0  # half-width of the square scene, metres
    relief: float = 2.0  # amplitude of the rolling terrain, metres
    n_ground_points: int = 1500
    n_buildings: int = 4
    points_per_building: int = 150
    building_size: tuple[float, float] = (4.0, 10.0)
    building_height: tuple[float, float] = (4.0, 12.0)


def terrain_height(xy: np.ndarray, relief: float, extent: float) -> np.ndarray:
    k = 2.0 * np.pi / extent
    return relief * (0.6 * np.sin(0.7 * k * xy[:, 0]) * np.cos(0.5 * k * xy[:, 1])
                     + 0.4 * np.sin(1.3 * k * xy[:, 1] + 0.5))


def make_terrain_points(spec: TerrainSpec | None = None, seed: int = 0) -> np.ndarray:
    """Point cloud of rolling terrain plus box-shaped buildings (tops and walls)."""
    spec = spec or TerrainSpec()
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-spec.extent, spec.extent, size=(spec.n_ground_points, 2))
    ground = np.column_stack([xy, terrain_height(xy, spec.relief, spec.extent)])
    parts = [ground]
    for _ in range(spec.n_buildings):
        w, d = rng.uniform(*spec.building_size, size=2)
        h = rng.uniform(*spec.building_height)
        cx, cy = rng.uniform(-0.6 * spec.extent, 0.6 * spec.extent, size=2)
        base = terrain_height(np.array([[cx, cy]]), spec.relief, spec.extent)[0]
        n_top = spec.points_per_building // 2
        top = np.column_stack([
            cx + rng.uniform(-w / 2, w / 2, n_top),
            cy + rng.uniform(-d / 2, d / 2, n_top),
            np.full(n_top, base + h),
        ])
        n_wall = spec.points_per_building - n_top
        s = rng.uniform(0, 2 * (w + d), n_wall)
        wx = np.where(s < w, cx - w / 2 + s, np.where(s < w + d, cx + w / 2,
                      np.where(s < 2 * w + d, cx + w / 2 - (s - w - d), cx - w / 2)))
        wy = np.where(s < w, cy - d / 2, np.where(s < w + d, cy - d / 2 + (s - w),
                      np.where(s < 2 * w + d, cy + d / 2, cy + d / 2 - (s - 2 * w - d))))
        walls = np.column_stack([wx, wy, base + rng.uniform(0, h, n_wall)])
        parts += [top, walls]
    return np.concatenate(parts)


def _project(camera: Camera, pose: Pose, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc = pose.transform(points)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack([camera.fx * pc[:, 0] / z + camera.cx,
                              camera.fy * pc[:, 1] / z + camera.cy])
    return uv, z


def make_scene(
    poses: list[Pose],
    points: np.ndarray,
    camera: Camera | None = None,
    pixel_noise: float = 0.0,
    min_track_length: int = 2,
    seed: int = 0,
) -> Reconstruction:
    """Ground-truth reconstruction: exact poses/points plus noisy 2D keypoint observations.

    Keypoints within each image are shuffled so keypoint index carries no information.
    Points seen by fewer than `min_track_length` views are dropped. There is no occlusion
    reasoning: points are infinitesimal.
    """
    camera = camera or default_camera()
    rng = np.random.default_rng(seed)
    recon = Reconstruction(cameras={1: camera})
    observations: dict[int, list[tuple[int, int]]] = {}
    per_image: list[tuple[np.ndarray, np.ndarray]] = []
    for image_idx, pose in enumerate(poses):
        uv, z = _project(camera, pose, points)
        visible = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
        pids = np.flatnonzero(visible)
        per_image.append((pids, uv[pids]))
        for pid in pids:
            observations.setdefault(int(pid), []).append(image_idx)
    keep = {pid for pid, views in observations.items() if len(views) >= min_track_length}

    for image_idx, (pids, uv) in enumerate(per_image):
        mask = np.array([int(p) in keep for p in pids], dtype=bool)
        pids, uv = pids[mask], uv[mask]
        order = rng.permutation(len(pids))
        pids, uv = pids[order], uv[order]
        uv = uv + rng.normal(0.0, pixel_noise, size=uv.shape) if pixel_noise > 0 else uv
        image_id = image_idx + 1
        recon.images[image_id] = Image(image_id, 1, f"synthetic_{image_id:04d}.png",
                                       pose=poses[image_idx], keypoints=uv)
        per_image[image_idx] = (pids, uv)

    tracks: dict[int, list[tuple[int, int]]] = {}
    for image_idx, (pids, _) in enumerate(per_image):
        for kp_idx, pid in enumerate(pids):
            tracks.setdefault(int(pid), []).append((image_idx + 1, kp_idx))
    for pid in sorted(tracks):
        recon.add_point(points[pid], tracks[pid])
    return recon


def correspondences(recon: Reconstruction, image_a: int, image_b: int) -> np.ndarray:
    """(M, 2) keypoint index pairs of ground-truth shared observations between two images."""
    a = recon.images[image_a].point3d_ids
    b = recon.images[image_b].point3d_ids
    index_b = {int(pid): k for k, pid in enumerate(b) if pid >= 0}
    pairs = [(k, index_b[int(pid)]) for k, pid in enumerate(a) if pid >= 0 and int(pid) in index_b]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def pairwise_matches(
    recon: Reconstruction,
    pairs: list[tuple[int, int]] | None = None,
    outlier_ratio: float = 0.0,
    min_matches: int = 20,
    seed: int = 0,
) -> dict[tuple[int, int], np.ndarray]:
    """Simulated feature matching output: GT correspondences contaminated with outliers.

    Returns {(image_a, image_b): (M, 2) keypoint index pairs} with image_a < image_b. This
    lets you test SfM independently of feature detection quality.
    """
    rng = np.random.default_rng(seed)
    ids = sorted(recon.images)
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    out = {}
    for a, b in pairs:
        inl = correspondences(recon, a, b)
        if len(inl) < min_matches:
            continue
        n_out = int(round(len(inl) * outlier_ratio / max(1.0 - outlier_ratio, 1e-9)))
        na, nb = len(recon.images[a].keypoints), len(recon.images[b].keypoints)
        bad = np.column_stack([rng.integers(0, na, n_out), rng.integers(0, nb, n_out)])
        m = np.concatenate([inl, bad])
        out[(a, b)] = m[rng.permutation(len(m))]
    return out


# ---------------------------------------------------------------------------------------
# Dense rendering for MVS: ground plane plus floating textured horizontal slabs.
# The scene is physically consistent (every ray hits the nearest surface), so rendered
# images and depth maps are exact ground truth for plane-sweep / PatchMatch experiments.
# ---------------------------------------------------------------------------------------


@dataclass
class Slab:
    """Horizontal textured rectangle at height z (a "roof")."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    z: float


def _texture(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = ndimage.gaussian_filter(rng.random((size, size)), sigma=1.5, mode="wrap")
    tex += 0.5 * ndimage.gaussian_filter(rng.random((size, size)), sigma=6.0, mode="wrap")
    tex -= tex.min()
    return tex / tex.max()


def render_slab_scene(
    camera: Camera,
    poses: list[Pose],
    slabs: list[Slab],
    ground_z: float = 0.0,
    texels_per_metre: float = 4.0,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Ray-cast grayscale images (float32 in [0, 1]) and z-depth maps for each pose.

    Pixel (row, col) is sampled at its center u = col + 0.5, v = row + 0.5 (COLMAP
    convention). Depth is camera-frame z; pixels that hit nothing get depth 0.
    """
    textures = [_texture(256, seed + i) for i in range(len(slabs) + 1)]
    cols, rows = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    rays_cam = np.stack([(cols + 0.5 - camera.cx) / camera.fx,
                         (rows + 0.5 - camera.cy) / camera.fy,
                         np.ones_like(cols, dtype=np.float64)], axis=-1)
    images, depths = [], []
    for pose in poses:
        d = rays_cam @ pose.R  # rows of R^T d_c, i.e. world-frame directions (z_cam = 1)
        c = pose.center
        best = np.full(cols.shape, np.inf)
        value = np.zeros(cols.shape)
        surfaces = [(ground_z, None)] + [(s.z, s) for s in slabs]
        for idx, (z, slab) in enumerate(surfaces):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (z - c[2]) / d[..., 2]
            x = c[0] + s * d[..., 0]
            y = c[1] + s * d[..., 1]
            hit = np.isfinite(s) & (s > 0) & (s < best)
            if slab is not None:
                hit &= (x >= slab.xmin) & (x <= slab.xmax) & (y >= slab.ymin) & (y <= slab.ymax)
            tex = textures[idx]
            coords = np.stack([y[hit] * texels_per_metre, x[hit] * texels_per_metre])
            value[hit] = ndimage.map_coordinates(tex, coords, order=1, mode="wrap")
            best[hit] = s[hit]
        depth = np.where(np.isfinite(best), best, 0.0)
        images.append(value.astype(np.float32))
        depths.append(depth.astype(np.float32))
    return images, depths
