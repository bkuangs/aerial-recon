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
from scipy.ndimage import uniform_filter

from aerial_recon.mvs.warp import warp_image
from aerial_recon.types import Camera, Pose


def relative_pose(ref_pose: Pose, src_pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with x_src = R x_ref + t, from two world-to-camera poses."""
    R = src_pose.R @ ref_pose.R.T  # noqa: N806
    return R, src_pose.t - R @ ref_pose.t


def plane_homography(
    K_ref: np.ndarray,  # noqa: N803
    K_src: np.ndarray,  # noqa: N803
    R: np.ndarray,  # noqa: N803
    t: np.ndarray,
    normal: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Homography mapping reference pixels to source pixels for the plane nᵀX = d."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return K_src @ (R + np.outer(t, n) / distance) @ np.linalg.inv(K_ref)


def _window_stats(a: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    mean = uniform_filter(a, window, mode="reflect")
    var = uniform_filter(a * a, window, mode="reflect") - mean * mean
    return mean, var


def _zncc_with_ref(ref_mean, ref_var, ref, b, window, eps):
    b_mean, b_var = _window_stats(b, window)
    cov = uniform_filter(ref * b, window, mode="reflect") - ref_mean * b_mean
    ok = (ref_var > eps) & (b_var > eps)
    out = np.zeros_like(cov)
    out[ok] = cov[ok] / np.sqrt(ref_var[ok] * b_var[ok])
    return np.clip(out, -1.0, 1.0)


def zncc(a: np.ndarray, b: np.ndarray, window: int = 7, eps: float = 1e-6) -> np.ndarray:
    """Windowed zero-mean normalized cross-correlation, per pixel, in [-1, 1].

    Compute local means/variances/covariance with a box filter
    (scipy.ndimage.uniform_filter or cv2.boxFilter). Where either window's variance is
    below `eps`, return 0 (textureless: no evidence either way).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_mean, a_var = _window_stats(a, window)
    return _zncc_with_ref(a_mean, a_var, a, b, window, eps)


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

    Choices made here:
      * A source that is invalid at a pixel (warps outside its image) is *excluded* from
        that pixel's mean; pixels with no valid source get score -1 (never win).
      * Sub-plane refinement is implemented: a parabola through the scores at the winner
        and its two neighbouring hypotheses, interpolated in hypothesis-index space
        (so it works for depth- or inverse-depth-uniform `depths`).
    """
    ref = np.asarray(ref_image, dtype=np.float32)
    h, w = ref.shape
    depths = np.asarray(depths, dtype=np.float64)
    n = np.array([0.0, 0.0, 1.0])
    eps = 1e-6
    ref_mean, ref_var = _window_stats(ref, window)
    rel = [relative_pose(ref_pose, p) for p in src_poses]
    srcs = [np.asarray(s, dtype=np.float32) for s in src_images]

    best = np.full((h, w), -np.inf, dtype=np.float32)
    best_idx = np.zeros((h, w), dtype=np.int32)
    left = np.full((h, w), np.nan, dtype=np.float32)  # score at best_idx - 1
    right = np.full((h, w), np.nan, dtype=np.float32)  # score at best_idx + 1
    prev = np.full((h, w), np.nan, dtype=np.float32)
    for k, d in enumerate(depths):
        total = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        for src, cam, (R, t) in zip(srcs, src_cameras, rel, strict=True):  # noqa: N806
            H = plane_homography(ref_camera.K, cam.K, R, t, n, d)  # noqa: N806
            warped, valid = warp_image(src, H, (h, w))
            score = _zncc_with_ref(ref_mean, ref_var, ref, warped, window, eps)
            total += np.where(valid, score, 0.0)
            count += valid
        score = np.where(count > 0, total / np.maximum(count, 1), -1.0).astype(np.float32)
        right = np.where(best_idx == k - 1, score, right)
        better = score > best
        best = np.where(better, score, best)
        best_idx = np.where(better, k, best_idx)
        left = np.where(better, prev, left)
        right = np.where(better, np.nan, right)
        prev = score

    offset = np.zeros((h, w), dtype=np.float64)
    ok = np.isfinite(left) & np.isfinite(right)
    denom = left - 2.0 * best + right
    ok &= denom < -1e-9
    offset[ok] = np.clip(0.5 * (left[ok] - right[ok]) / denom[ok], -0.5, 0.5)
    pos = np.clip(best_idx + offset, 0, len(depths) - 1)
    depth = np.interp(pos, np.arange(len(depths)), depths)
    return depth.astype(np.float32), best.astype(np.float32)
