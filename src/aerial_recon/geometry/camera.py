"""M1 — The pinhole camera with radial distortion (COLMAP RADIAL model).

    x_cam = R x_world + t                      (world -> camera, see types.Pose)
    (x, y) = (X/Z, Y/Z)                        (normalized image plane)
    r² = x² + y²;  d = 1 + k1 r² + k2 r⁴       (radial distortion)
    u = fx · d·x + cx,  v = fy · d·y + cy      (pixels; top-left pixel center is (0.5, 0.5))

Drone cameras (DJI etc.) have noticeable barrel distortion; ignoring it bends straight
roads and makes long survey flights "bowl" (the famous doming effect).

Reading: Hartley & Zisserman (HZ) ch. 6; Szeliski 2e §2.1.5 and §11.1.
"""

from __future__ import annotations

import numpy as np

from aerial_recon.types import Camera, Pose


def distort(camera: Camera, xy: np.ndarray) -> np.ndarray:
    """Apply radial distortion to normalized coordinates (N, 2) -> distorted normalized (N, 2)."""
    r2 = np.sum(xy**2, axis=1, keepdims=True)
    d = 1 + camera.k1 * r2 + camera.k2 * r2**2
    return xy * d


def undistort(camera: Camera, xy_distorted: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Invert `distort` numerically (fixed-point iteration or Newton). (N, 2) -> (N, 2)."""
    # Fixed point of xy = xy_d / d(xy); converges fast for the mild distortion of real lenses.
    xy = xy_distorted.copy()
    for _ in range(iterations):
        r2 = np.sum(xy**2, axis=1, keepdims=True)
        d = 1 + camera.k1 * r2 + camera.k2 * r2**2
        xy = xy_distorted / d
    return xy


def project(camera: Camera, pose: Pose, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points (N, 3) to pixels.

    Returns (uv (N, 2), depth (N,)) where depth is camera-frame Z. Points behind the camera
    still get (meaningless) pixel values; callers filter with depth > 0.
    """
    x_cam = points_world @ pose.R.T + pose.t

    xy = x_cam[:, :2] / x_cam[:, 2:]
    xy_d = distort(camera, xy)

    uv = np.column_stack([camera.fx * xy_d[:, 0] + camera.cx,
                        camera.fy * xy_d[:, 1] + camera.cy])

    return uv, x_cam[:, 2]


def unproject(camera: Camera, pose: Pose, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) with camera-frame Z depth (N,) -> world points (N, 3). Inverse of project."""
    xy = pixel_to_normalized(camera, uv)
    x_cam = np.column_stack([xy * depth[:, None], depth])
    # Row-vector form of R^T (x_cam - t)
    return (x_cam - pose.t) @ pose.R


def pixel_to_normalized(camera: Camera, uv: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) -> undistorted normalized image coordinates (N, 2), i.e. K⁻¹ then undistort.

    Epipolar geometry and PnP operate in this space.
    """
    xy_d = (uv - [camera.cx, camera.cy]) / [camera.fx, camera.fy]
    return undistort(camera, xy_d)


def reprojection_errors(
    camera: Camera, pose: Pose, points_world: np.ndarray, uv_observed: np.ndarray
) -> np.ndarray:
    """Euclidean pixel distance between projected points and observations, shape (N,)."""
    uv_projected, _ = project(camera, pose, points_world)

    return np.linalg.norm(uv_projected - uv_observed, axis=1)
