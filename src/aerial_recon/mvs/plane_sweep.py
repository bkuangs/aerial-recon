"""M8 — Plane-sweep multi-view stereo.

For each hypothesized plane, warp every source image into the reference view through the
plane-induced homography and measure photo-consistency (ZNCC over a window). The best
plane per pixel gives its depth. Fronto-parallel planes in the reference camera are the
textbook choice; for a nadir drone they coincide with horizontal ground planes, which is
why plane sweep works so well on aerial imagery (and why building facades break it).

    H = K_src (R + t nᵀ / d) K_ref⁻¹     for the plane nᵀX = d in reference-camera coords,
                                         where x_src = R x_ref + t.

Reading: Collins "A Space-Sweep Approach to True Multi-Image Matching" (CVPR 1996);
Gallup et al. "Real-Time Plane-Sweeping Stereo with Multiple Sweeping Directions" (2007);
Furukawa & Hernández "Multi-View Stereo: A Tutorial" (2015) ch. 2–3; Szeliski 2e §12.1–12.7.
"""

from __future__ import annotations

import numpy as np

from aerial_recon.types import Camera, Pose


def relative_pose(ref_pose: Pose, src_pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with x_src = R x_ref + t, from two world-to-camera poses."""
    raise NotImplementedError("M8: implement relative_pose")


def plane_homography(
    K_ref: np.ndarray,  # noqa: N803
    K_src: np.ndarray,  # noqa: N803
    R: np.ndarray,  # noqa: N803
    t: np.ndarray,
    normal: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Homography mapping reference pixels to source pixels for the plane nᵀX = d."""
    raise NotImplementedError("M8: implement plane_homography")


def zncc(a: np.ndarray, b: np.ndarray, window: int = 7, eps: float = 1e-6) -> np.ndarray:
    """Windowed zero-mean normalized cross-correlation, per pixel, in [-1, 1].

    Compute local means/variances/covariance with a box filter
    (scipy.ndimage.uniform_filter or cv2.boxFilter). Where either window's variance is
    below `eps`, return 0 (textureless: no evidence either way).
    """
    raise NotImplementedError("M8: implement zncc")


def plane_sweep_depth(
    ref_image: np.ndarray,
    ref_camera: Camera,
    ref_pose: Pose,
    src_images: list[np.ndarray],
    src_cameras: list[Camera],
    src_poses: list[Pose],
    depths: np.ndarray,
    window: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Fronto-parallel plane sweep over `depths` (D,) for a grayscale reference image.

    For each depth: warp each source (mvs.warp.warp_image), score with zncc, and average
    over the sources that are valid at that pixel (invalid -> treat score as -1 or
    exclude; document your choice). Winner-takes-all over depths.

    Returns (depth (H, W), score (H, W)) where score is the winning mean ZNCC. Images must
    be undistorted (distortion k1 = k2 = 0).

    Stretch: sub-plane refinement by fitting a parabola to the cost around the winner;
    sweeping world-horizontal planes for oblique views.
    """
    raise NotImplementedError("M8: implement plane_sweep_depth")
