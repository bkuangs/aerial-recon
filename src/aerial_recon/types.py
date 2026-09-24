"""Core data contracts shared by every stage of the pipeline.

Conventions (see docs/conventions.md):
  * Camera frame is OpenCV/COLMAP: +x right, +y down, +z forward (optical axis).
  * Poses are world-to-camera: x_cam = R @ x_world + t.
  * Pixel coordinates follow COLMAP: the *center* of the top-left pixel is (0.5, 0.5).
  * World frame for aerial scenes is local ENU (x east, y north, z up) once georeferenced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics with optional two-term radial distortion (COLMAP RADIAL model)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0

    @property
    def K(self) -> np.ndarray:  # noqa: N802 - standard notation
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def has_distortion(self) -> bool:
        return self.k1 != 0.0 or self.k2 != 0.0

    def scaled(self, factor: float) -> Camera:
        """Intrinsics for an image resized by `factor` (e.g. 0.5 for half resolution)."""
        return Camera(
            width=round(self.width * factor),
            height=round(self.height * factor),
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            k1=self.k1,
            k2=self.k2,
        )

    @staticmethod
    def from_fov(width: int, height: int, hfov_deg: float) -> Camera:
        fx = 0.5 * width / np.tan(0.5 * np.deg2rad(hfov_deg))
        return Camera(width, height, fx, fx, width / 2.0, height / 2.0)


@dataclass(frozen=True)
class Pose:
    """Rigid world-to-camera transform: x_cam = R @ x_world + t."""

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", np.asarray(self.R, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64).reshape(3))

    @staticmethod
    def identity() -> Pose:
        return Pose(np.eye(3), np.zeros(3))

    @staticmethod
    def from_center(R: np.ndarray, center: np.ndarray) -> Pose:  # noqa: N803
        R = np.asarray(R, dtype=np.float64)
        return Pose(R, -R @ np.asarray(center, dtype=np.float64))

    @staticmethod
    def from_matrix(T: np.ndarray) -> Pose:  # noqa: N803
        return Pose(T[:3, :3], T[:3, 3])

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates, C = -R^T t."""
        return -self.R.T @ self.t

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def projection_matrix(self, K: np.ndarray) -> np.ndarray:  # noqa: N803
        """3x4 P = K [R | t]."""
        return K @ np.hstack([self.R, self.t[:, None]])

    def inverse(self) -> Pose:
        return Pose(self.R.T, -self.R.T @ self.t)

    def compose(self, other: Pose) -> Pose:
        """self ∘ other: apply `other` first, then `self`."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def transform(self, points_world: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> camera frame (N, 3)."""
        return points_world @ self.R.T + self.t


@dataclass
class Image:
    """A registered (or not yet registered) view with its 2D keypoints."""

    image_id: int
    camera_id: int
    name: str
    pose: Pose | None = None
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3d_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64).reshape(-1, 2)
        if self.point3d_ids.shape[0] != self.keypoints.shape[0]:
            self.point3d_ids = -np.ones(self.keypoints.shape[0], dtype=np.int64)

    @property
    def is_registered(self) -> bool:
        return self.pose is not None


@dataclass
class Point3D:
    xyz: np.ndarray
    rgb: np.ndarray = field(default_factory=lambda: np.array([128, 128, 128], dtype=np.uint8))
    track: list[tuple[int, int]] = field(default_factory=list)  # (image_id, keypoint_index)
    error: float = 0.0


@dataclass
class Reconstruction:
    """Sparse scene: cameras, posed images with keypoints, and 3D points with tracks.

    Mirrors COLMAP's data model so it round-trips through io.colmap_text.
    """

    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, Image] = field(default_factory=dict)
    points: dict[int, Point3D] = field(default_factory=dict)

    @property
    def registered_image_ids(self) -> list[int]:
        return sorted(i for i, im in self.images.items() if im.is_registered)

    def points_array(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 3))
        return np.stack([p.xyz for p in self.points.values()])

    def centers(self, image_ids: list[int] | None = None) -> np.ndarray:
        ids = self.registered_image_ids if image_ids is None else image_ids
        return np.stack([self.images[i].pose.center for i in ids])

    def next_point_id(self) -> int:
        return max(self.points, default=0) + 1

    def add_point(self, xyz: np.ndarray, track: list[tuple[int, int]], rgb=None) -> int:
        pid = self.next_point_id()
        point = Point3D(np.asarray(xyz, dtype=np.float64), track=list(track))
        if rgb is not None:
            point.rgb = np.asarray(rgb, dtype=np.uint8)
        self.points[pid] = point
        for image_id, kp_idx in track:
            self.images[image_id].point3d_ids[kp_idx] = pid
        return pid

    def summary(self) -> str:
        n_obs = sum(len(p.track) for p in self.points.values())
        mean_track = n_obs / max(len(self.points), 1)
        return (
            f"{len(self.cameras)} cameras, {len(self.registered_image_ids)}/{len(self.images)} "
            f"registered images, {len(self.points)} points, mean track length {mean_track:.2f}"
        )
