"""M6 — Incremental Structure-from-Motion (a mini COLMAP).

    verify pairs (relative pose RANSAC)  ->  choose initial pair  ->  triangulate
    loop:  pick next image  ->  PnP register  ->  triangulate new tracks  ->  filter
           -> local/global bundle adjustment every few images
    final global bundle adjustment

The public contract is just `IncrementalSfM(...).run() -> Reconstruction`; the internal
methods below are a suggested decomposition. Change them freely.

Design questions to answer in your notes as you go:
  * Initial pair: many inliers *and* a wide baseline (large median triangulation angle,
    low homography inlier ratio). Why does the pair with the most matches often fail?
  * Next image: most 2D-3D correspondences? Best spatial coverage in the image
    (COLMAP's visibility pyramid)?
  * When a track has an outlier observation, do you drop the observation or the point?
  * A few outlier matches survive epipolar verification (they lie near the epipolar line)
    and merge two tracks into one "conflicting" track with two keypoints in the same
    image. Dropping such tracks is simple but loses long, valuable tracks. Can you split
    them instead? (The M6 gate only requires 30% of GT points; the reference drops.)

Reading: Schönberger & Frahm "Structure-from-Motion Revisited" (CVPR 2016) — read it
twice, once now and once after your implementation works; Snavely et al. "Photo Tourism"
(SIGGRAPH 2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerial_recon.types import Camera, Image, Reconstruction


@dataclass
class SfMOptions:
    # "eight_point": your M4 estimator. "five_point_opencv": cv2.findEssentialMat +
    # cv2.recoverPose. Run the real-data gate with both and explain the gap (hint: flat
    # terrain makes F degenerate; the calibrated 5-point E does not care).
    relative_pose_solver: str = "eight_point"
    min_pair_inliers: int = 30
    relative_pose_threshold_px: float = 2.0
    pnp_threshold_px: float = 4.0
    min_pnp_inliers: int = 20
    min_triangulation_angle_deg: float = 1.5
    max_reprojection_error_px: float = 4.0
    global_ba_every: int = 5
    seed: int = 0


class IncrementalSfM:
    def __init__(
        self,
        cameras: dict[int, Camera],
        images: dict[int, Image],
        matches: dict[tuple[int, int], np.ndarray],
        options: SfMOptions | None = None,
    ) -> None:
        """
        Args:
            cameras: intrinsics by camera_id (assumed known and fixed; undistort first or
                make sure every geometry call handles distortion).
            images: unregistered images (pose=None) with keypoints filled in.
            matches: raw putative matches {(a, b): (M, 2)} with a < b, outliers included.
        """
        self.cameras = cameras
        self.images = images
        self.matches = matches
        self.options = options or SfMOptions()
        self.rng = np.random.default_rng(self.options.seed)

    def run(self) -> Reconstruction:
        """Reconstruct and return the largest model found."""
        raise NotImplementedError("M6: implement IncrementalSfM.run")

    # Suggested steps --------------------------------------------------------------------

    def verify_pairs(self) -> None:
        """Relative pose RANSAC per pair; keep inlier matches for pairs with enough support."""
        raise NotImplementedError

    def choose_initial_pair(self) -> tuple[int, int]:
        raise NotImplementedError

    def next_image(self) -> int | None:
        raise NotImplementedError

    def register_image(self, image_id: int) -> bool:
        raise NotImplementedError

    def triangulate_image(self, image_id: int) -> int:
        raise NotImplementedError

    def filter_points(self) -> int:
        raise NotImplementedError
