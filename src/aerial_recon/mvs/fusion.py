"""M9 — Depth-map fusion: from per-view depth maps to one consistent point cloud.

Individual depth maps are noisy and wrong in occluded / textureless regions. A depth is
trusted only if other views agree: project the pixel into a source view, read the source
depth there, lift it back to 3D, and reproject into the reference. If it lands close in
pixels and in depth for enough sources, keep it.

Reading: Merrell et al. "Real-Time Visibility-Based Fusion of Depth Maps" (ICCV 2007);
Schönberger et al. (ECCV 2016) §4.4 "Fusion"; Yao et al. "MVSNet" (ECCV 2018) §4.2 for
the same check in the learned-MVS world.
"""

from __future__ import annotations

import numpy as np

from aerial_recon.geometry.camera import project, unproject
from aerial_recon.types import Camera, Pose


def backproject_depth(
    depth: np.ndarray, camera: Camera, pose: Pose
) -> tuple[np.ndarray, np.ndarray]:
    """World points for every pixel with depth > 0, sampled at pixel centers (col+0.5, row+0.5).

    Returns (points (N, 3), flat_indices (N,)) with flat_indices into depth.ravel(),
    row-major order.
    """
    depth = np.asarray(depth)
    flat = np.flatnonzero(depth.ravel() > 0)
    rows, cols = np.divmod(flat, depth.shape[1])
    uv = np.column_stack([cols + 0.5, rows + 0.5]).astype(np.float64)
    pts = unproject(camera, pose, uv, depth.ravel()[flat].astype(np.float64))
    return pts, flat


def _consistent_count(ref_depth, ref_camera, ref_pose, src_depths, src_cameras, src_poses,
                      max_reprojection_px, max_relative_depth):
    h, w = ref_depth.shape
    pts, flat = backproject_depth(ref_depth, ref_camera, ref_pose)
    rows, cols = np.divmod(flat, w)
    p_ref = np.column_stack([cols + 0.5, rows + 0.5])
    z_ref = ref_depth.ravel()[flat].astype(np.float64)
    count = np.zeros(len(flat), dtype=np.int32)
    for sd, sc, sp in zip(src_depths, src_cameras, src_poses, strict=True):
        uv, z = project(sc, sp, pts)
        c = np.floor(uv[:, 0]).astype(np.int64)
        r = np.floor(uv[:, 1]).astype(np.int64)
        inb = (z > 0) & (c >= 0) & (c < sd.shape[1]) & (r >= 0) & (r < sd.shape[0])
        idx = np.flatnonzero(inb)
        d_src = sd[r[idx], c[idx]].astype(np.float64)
        has = d_src > 0
        idx, d_src = idx[has], d_src[has]
        if len(idx) == 0:
            continue
        uv_src = np.column_stack([c[idx] + 0.5, r[idx] + 0.5]).astype(np.float64)
        back = unproject(sc, sp, uv_src, d_src)
        uv_back, z_back = project(ref_camera, ref_pose, back)
        ok = (np.linalg.norm(uv_back - p_ref[idx], axis=1) < max_reprojection_px) \
            & (np.abs(z_back - z_ref[idx]) / z_ref[idx] < max_relative_depth)
        count[idx[ok]] += 1
    return count, flat


def consistency_mask(
    ref_depth: np.ndarray,
    ref_camera: Camera,
    ref_pose: Pose,
    src_depths: list[np.ndarray],
    src_cameras: list[Camera],
    src_poses: list[Pose],
    max_reprojection_px: float = 1.0,
    max_relative_depth: float = 0.01,
    min_consistent: int = 1,
) -> np.ndarray:
    """(H, W) bool: reference pixels whose depth is confirmed by >= min_consistent sources.

    For a source: project ref pixel p (at its depth) into the source; sample the source
    depth at the nearest pixel; lift that source pixel to 3D and project back into the ref
    to get p' and depth z'. Consistent iff |p - p'| < max_reprojection_px and
    |z' - z| / z < max_relative_depth. Pixels with depth <= 0 are never consistent.
    """
    count, flat = _consistent_count(np.asarray(ref_depth), ref_camera, ref_pose,
                                    [np.asarray(d) for d in src_depths], src_cameras,
                                    src_poses, max_reprojection_px, max_relative_depth)
    mask = np.zeros(np.asarray(ref_depth).size, dtype=bool)
    mask[flat[count >= min_consistent]] = True
    return mask.reshape(np.asarray(ref_depth).shape)


def fuse_depth_maps(
    depths: list[np.ndarray],
    cameras: list[Camera],
    poses: list[Pose],
    colors: list[np.ndarray] | None = None,
    min_consistent: int = 1,
    max_reprojection_px: float = 1.0,
    max_relative_depth: float = 0.01,
    neighbors: list[list[int]] | None = None,
    voxel_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Filter every view against all others and concatenate the surviving points.

    Returns (points (N, 3), colors (N, C) or None). Stretch: average consistent
    observations instead of keeping duplicates, and voxel-downsample.

    `neighbors[i]` optionally restricts view i's check to those source views (large
    scenes: all-pairs is quadratic). `voxel_size` averages points (and colors) per voxel.
    """
    acc = _VoxelAccumulator(voxel_size) if voxel_size is not None else None
    all_pts, all_cols = [], []
    for i, depth in enumerate(depths):
        srcs = neighbors[i] if neighbors is not None else [j for j in range(len(depths))
                                                           if j != i]
        mask = consistency_mask(depth, cameras[i], poses[i], [depths[j] for j in srcs],
                                [cameras[j] for j in srcs], [poses[j] for j in srcs],
                                max_reprojection_px, max_relative_depth, min_consistent)
        pts, flat = backproject_depth(np.where(mask, depth, 0), cameras[i], poses[i])
        cols = None
        if colors is not None:
            c = np.asarray(colors[i])
            cols = c.reshape(c.shape[0] * c.shape[1], -1)[flat]
        if acc is not None:
            acc.add(pts, cols)
        else:
            all_pts.append(pts)
            if cols is not None:
                all_cols.append(cols)
    if acc is not None:
        return acc.result()
    points = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    out_cols = np.concatenate(all_cols) if colors is not None and all_cols else None
    return points, out_cols


_OFF = 1 << 20  # voxel index offset: +-1M voxels per axis


_MASK = (1 << 21) - 1


def _voxel_keys_from_index(idx: np.ndarray) -> np.ndarray:
    idx = idx + _OFF
    return (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]


class _VoxelAccumulator:
    """Streaming per-voxel sums so large scenes never hold every raw point in memory.

    Each view is reduced on its own and buffered; buffers are merged into the sorted store
    only when they grow large, so the store is not re-sorted once per view. Sums are kept
    relative to the voxel corner in float32 (sub-voxel offsets, so precision is ample),
    which with int64 keys and int32 counts costs 32 bytes per voxel (+12 with colors).
    """

    def __init__(self, voxel_size: float, flush_points: int = 4_000_000) -> None:
        self.voxel_size = voxel_size
        self.flush_points = flush_points
        self.keys = np.zeros(0, dtype=np.int64)
        self.pts = np.zeros((0, 3), dtype=np.float32)
        self.cols: np.ndarray | None = None
        self.counts = np.zeros(0, dtype=np.int32)
        self._buffer: list[tuple] = []
        self._buffered = 0
        self._has_colors: bool | None = None

    def add(self, points: np.ndarray, colors: np.ndarray | None) -> None:
        if len(points) == 0:
            return
        if self._has_colors is None:
            self._has_colors = colors is not None
        idx = np.floor(points / self.voxel_size).astype(np.int64)
        offsets = (points - idx * self.voxel_size).astype(np.float32)
        keys = _voxel_keys_from_index(idx)
        del idx
        cols = None if colors is None else np.asarray(colors, dtype=np.float32)
        red = self._reduce(keys, offsets, cols, np.ones(len(keys), dtype=np.int32))
        self._buffer.append(red)
        self._buffered += len(red[0])
        if self._buffered >= max(self.flush_points, len(self.keys) // 4):
            self._flush()

    @staticmethod
    def _reduce(keys, pts, cols, counts):
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        uniq = keys[starts]
        pts = np.add.reduceat(pts[order], starts, axis=0, dtype=np.float32)
        if cols is not None:
            cols = np.add.reduceat(cols[order], starts, axis=0, dtype=np.float32)
        counts = np.add.reduceat(counts[order], starts).astype(np.int32)
        return uniq, pts, cols, counts

    def _flush(self) -> None:
        if not self._buffer:
            return
        parts = self._buffer
        self._buffer, self._buffered = [], 0
        keys = np.concatenate([b[0] for b in parts])
        pts = np.concatenate([b[1] for b in parts])
        cols = np.concatenate([b[2] for b in parts]) if self._has_colors else None
        counts = np.concatenate([b[3] for b in parts])
        del parts
        keys, pts, cols, counts = self._reduce(keys, pts, cols, counts)
        if self._has_colors and self.cols is None:
            self.cols = np.zeros((0, cols.shape[1]), dtype=np.float32)
        # Merge into the sorted store: add in place where the voxel exists, insert the rest.
        pos = np.searchsorted(self.keys, keys)
        hit = np.zeros(len(keys), dtype=bool)
        inside = pos < len(self.keys)
        hit[inside] = self.keys[pos[inside]] == keys[inside]
        at = pos[hit]
        self.pts[at] += pts[hit]
        self.counts[at] += counts[hit]
        if cols is not None:
            self.cols[at] += cols[hit]
        miss = ~hit
        where = pos[miss]
        self.keys = np.insert(self.keys, where, keys[miss])
        self.pts = np.insert(self.pts, where, pts[miss], axis=0)
        self.counts = np.insert(self.counts, where, counts[miss])
        if cols is not None:
            self.cols = np.insert(self.cols, where, cols[miss], axis=0)

    def result(self) -> tuple[np.ndarray, np.ndarray | None]:
        self._flush()
        if len(self.keys) == 0:
            return np.zeros((0, 3)), None
        n = self.counts[:, None].astype(np.float64)
        idx = np.column_stack([(self.keys >> 42) & _MASK, (self.keys >> 21) & _MASK,
                               self.keys & _MASK]) - _OFF
        pts = idx * self.voxel_size + self.pts / n
        return pts, (None if self.cols is None else self.cols / n)


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray | None, voxel_size: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Average points (and colors) that fall in the same cubic voxel."""
    acc = _VoxelAccumulator(voxel_size)
    acc.add(np.asarray(points, dtype=np.float64),
            None if colors is None else np.asarray(colors, dtype=np.float64).reshape(
                len(points), -1))
    return acc.result()
