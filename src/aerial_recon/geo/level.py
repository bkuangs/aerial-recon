"""Gravity-level a model without GPS (video frames carry no EXIF).

A gimballed drone camera has (almost) zero roll, so every camera x-axis is horizontal.
World "up" is therefore the direction most orthogonal to all camera x-axes: the smallest
eigenvector of sum x_i x_iᵀ, signed so the cameras are above the scene. That is well
conditioned whenever the heading varies through more than a line (orbits). For straight
or back-and-forth flights the x-axes are (anti)parallel, and we fall back to the mean
scene -> camera direction made orthogonal to the common x-axis.

The result is only a rotation + translation (+ an optional user scale): the model stays in
arbitrary SfM units, but z is up and the origin sits at the median sparse point.
"""

from __future__ import annotations

import numpy as np

from aerial_recon.geometry.alignment import Sim3, transform_reconstruction
from aerial_recon.types import Reconstruction


def estimate_up(recon: Reconstruction) -> np.ndarray:
    """Unit world up vector from registered camera orientations."""
    poses = [im.pose for im in recon.images.values() if im.pose is not None]
    x_axes = np.stack([p.R[0] for p in poses])  # camera x in world = first row of R (w2c)
    scene = np.median(np.stack([p.xyz for p in recon.points.values()]), axis=0)
    above = np.mean([p.center - scene for p in poses], axis=0)  # cameras fly above the scene
    evals, evecs = np.linalg.eigh(x_axes.T @ x_axes)
    if evals[1] > 0.05 * evals[2]:
        up = evecs[:, 0]
    else:
        x_mean = evecs[:, 2]
        up = above - (above @ x_mean) * x_mean
    up /= np.linalg.norm(up)
    return up if up @ above > 0 else -up


def _rotation_to_z(up: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(up @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def level_reconstruction(recon: Reconstruction, scale: float = 1.0) -> dict:
    """In place: rotate so the estimated up is +z, move the median sparse point to the
    origin, multiply by `scale`. Returns a summary."""
    up = estimate_up(recon)
    R = _rotation_to_z(up)  # noqa: N806
    xyz = np.stack([p.xyz for p in recon.points.values()])
    origin = np.median(xyz @ R.T, axis=0)
    sim3 = Sim3(scale, R, -scale * origin)
    transform_reconstruction(recon, sim3)
    centers = np.stack([im.pose.center for im in recon.images.values() if im.pose is not None])
    ground = np.median(np.stack([p.xyz for p in recon.points.values()])[:, 2])
    tilt = []
    for im in recon.images.values():
        if im.pose is not None:
            roll = np.degrees(np.arcsin(np.clip(im.pose.R[0] @ np.array([0, 0, 1.0]), -1, 1)))
            tilt.append(abs(roll))
    return {
        "up_before": up.tolist(),
        "scale": scale,
        "num_images": len(centers),
        "median_camera_height": float(np.median(centers[:, 2]) - ground),
        "camera_height_std": float(np.std(centers[:, 2])),
        "median_abs_roll_deg": float(np.median(tilt)),
        "max_abs_roll_deg": float(np.max(tilt)),
    }
