"""M8 stretch — PatchMatch stereo with per-pixel slanted planes.

Plane sweep tests a fixed set of planes for all pixels. PatchMatch gives every pixel its
own plane (depth + normal), initialized randomly, and improves it by (1) propagating good
planes from neighbours and (2) random refinement. It handles slanted surfaces (roofs,
facades) and is what COLMAP's dense stereo does (plus pixelwise view selection).

No tests: validate it with the same synthetic slab scene as plane sweep, then on ETH3D.

Reading: Bleyer et al. "PatchMatch Stereo" (BMVC 2011); Galliani et al. "Gipuma" (ICCV
2015); Schönberger et al. "Pixelwise View Selection for Unstructured MVS" (ECCV 2016).
"""

from __future__ import annotations

import numpy as np

from aerial_recon.types import Camera, Pose


def patchmatch_depth(
    ref_image: np.ndarray,
    ref_camera: Camera,
    ref_pose: Pose,
    src_images: list[np.ndarray],
    src_cameras: list[Camera],
    src_poses: list[Pose],
    depth_range: tuple[float, float],
    iterations: int = 4,
    window: int = 7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (depth (H, W), normal (H, W, 3) in ref camera frame, score (H, W))."""
    raise NotImplementedError("M8 stretch: implement patchmatch_depth")
