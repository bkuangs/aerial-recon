"""M6 — Perspective-n-Point: registering a new image against the existing map.

Incremental SfM grows the model one image at a time: find 2D-3D matches between the new
image's keypoints and already triangulated points, then solve for the camera pose.

Reading: HZ §7.1–7.2 (DLT camera resection); Lepetit et al. "EPnP" (IJCV 2009);
Kneip et al. "P3P" (CVPR 2011). OpenCV's solvePnPRansac is a fair reference to compare to.
"""

from __future__ import annotations

import numpy as np

from aerial_recon.types import Camera, Pose


def pnp_dlt(points_world: np.ndarray, xn: np.ndarray) -> Pose | None:
    """Calibrated DLT resection from N >= 6 correspondences.

    xn are *normalized* image coordinates (N, 2) (see camera.pixel_to_normalized). Solve the
    homogeneous 12-unknown system for P = [R | t] up to scale, then fix scale and sign so
    that R is a proper rotation (project_to_so3) and points have positive depth.
    Normalize the 3D points (centroid/scale) for conditioning. Returns None if degenerate.
    """
    raise NotImplementedError("M6: implement pnp_dlt")


def pnp_ransac(
    points_world: np.ndarray,
    uv: np.ndarray,
    camera: Camera,
    threshold_px: float = 4.0,
    confidence: float = 0.999,
    rng: np.random.Generator | None = None,
) -> tuple[Pose | None, np.ndarray]:
    """RANSAC (M3) around pnp_dlt with pixel reprojection error, sample size 6.

    Returns (pose, inliers). Follow with motion-only bundle adjustment for accuracy.
    """
    raise NotImplementedError("M6: implement pnp_ransac")
