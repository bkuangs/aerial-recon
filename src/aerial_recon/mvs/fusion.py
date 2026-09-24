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

from aerial_recon.types import Camera, Pose


def backproject_depth(
    depth: np.ndarray, camera: Camera, pose: Pose
) -> tuple[np.ndarray, np.ndarray]:
    """World points for every pixel with depth > 0, sampled at pixel centers (col+0.5, row+0.5).

    Returns (points (N, 3), flat_indices (N,)) with flat_indices into depth.ravel(),
    row-major order.
    """
    raise NotImplementedError("M9: implement backproject_depth")


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
    raise NotImplementedError("M9: implement consistency_mask")


def fuse_depth_maps(
    depths: list[np.ndarray],
    cameras: list[Camera],
    poses: list[Pose],
    colors: list[np.ndarray] | None = None,
    min_consistent: int = 1,
    max_reprojection_px: float = 1.0,
    max_relative_depth: float = 0.01,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Filter every view against all others and concatenate the surviving points.

    Returns (points (N, 3), colors (N, C) or None). Stretch: average consistent
    observations instead of keeping duplicates, and voxel-downsample.
    """
    raise NotImplementedError("M9: implement fuse_depth_maps")
