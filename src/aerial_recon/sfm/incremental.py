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

import time
from dataclasses import dataclass

import cv2
import numpy as np

from aerial_recon.geometry.camera import pixel_to_normalized, project
from aerial_recon.geometry.epipolar import estimate_relative_pose
from aerial_recon.geometry.pnp import pnp_ransac
from aerial_recon.geometry.triangulation import triangulate_dlt, triangulation_angles_deg
from aerial_recon.sfm.bundle_adjustment import bundle_adjust
from aerial_recon.sfm.tracks import build_tracks
from aerial_recon.types import Camera, Image, Pose, Reconstruction


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
    # Initial pair: prefer pairs whose median inlier triangulation angle is at least this.
    init_min_triangulation_angle_deg: float = 4.0
    # Try this many initial pairs and keep the model with the most registered images.
    max_init_attempts: int = 3
    verbose: bool = False


@dataclass
class PairGeometry:
    matches: np.ndarray  # (M, 2) verified inlier keypoint index pairs
    R: np.ndarray  # x_b = R x_a + t (unit t)
    t: np.ndarray
    median_angle_deg: float


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
        # Undistorted normalized coordinates of every keypoint: all geometry happens here,
        # so radial distortion is handled once instead of in every estimator.
        self.xn = {iid: pixel_to_normalized(cameras[im.camera_id], im.keypoints)
                   for iid, im in images.items()}
        self.pairs: dict[tuple[int, int], PairGeometry] = {}
        self.tracks: list[list[tuple[int, int]]] = []
        self.kp_track: dict[int, np.ndarray] = {}

    def _log(self, msg: str) -> None:
        if self.options.verbose:
            print(msg, flush=True)

    def run(self) -> Reconstruction:
        """Reconstruct and return the largest model found."""
        t0 = time.time()
        self.verify_pairs()
        self._log(f"verified {len(self.pairs)}/{len(self.matches)} pairs "
                  f"({time.time() - t0:.1f}s)")
        self.tracks = build_tracks({p: g.matches for p, g in self.pairs.items()})
        self.kp_track = {iid: -np.ones(len(im.keypoints), dtype=np.int64)
                         for iid, im in self.images.items()}
        for tid, track in enumerate(self.tracks):
            for image_id, kp_idx in track:
                self.kp_track[image_id][kp_idx] = tid
        self._log(f"{len(self.tracks)} tracks")

        best: Reconstruction | None = None
        tried: set[tuple[int, int]] = set()
        for _ in range(self.options.max_init_attempts):
            pair = self.choose_initial_pair(exclude=tried)
            if pair is None:
                break
            tried.add(pair)
            recon = self._reconstruct(pair)
            self._log(f"init {pair}: {recon.summary()}")
            if best is None or len(recon.registered_image_ids) > len(best.registered_image_ids):
                best = recon
            if len(best.registered_image_ids) == len(self.images):
                break
        if best is None:
            best = Reconstruction(cameras=dict(self.cameras), images=self._fresh_images())
        self._log(f"SfM done in {time.time() - t0:.1f}s")
        return best

    # Suggested steps --------------------------------------------------------------------

    def verify_pairs(self) -> None:
        """Relative pose RANSAC per pair; keep inlier matches for pairs with enough support."""
        opt = self.options
        cv2.setRNGSeed(opt.seed)
        for (a, b), m in sorted(self.matches.items()):
            m = np.asarray(m, dtype=np.int64).reshape(-1, 2)
            if len(m) < max(opt.min_pair_inliers, 8):
                continue
            cam_a = self.cameras[self.images[a].camera_id]
            cam_b = self.cameras[self.images[b].camera_id]
            xa, xb = self.xn[a][m[:, 0]], self.xn[b][m[:, 1]]
            if opt.relative_pose_solver == "eight_point":
                Ka, Kb = cam_a.K, cam_b.K  # noqa: N806
                ua = xa * [cam_a.fx, cam_a.fy] + [cam_a.cx, cam_a.cy]
                ub = xb * [cam_b.fx, cam_b.fy] + [cam_b.cx, cam_b.cy]
                rel = estimate_relative_pose(ua, ub, Ka, Kb,
                                             threshold_px=opt.relative_pose_threshold_px,
                                             rng=self.rng)
                if rel.F is None:
                    continue
                R, t, inliers = rel.R, rel.t, rel.inliers  # noqa: N806
            elif opt.relative_pose_solver == "five_point_opencv":
                f = 0.25 * (cam_a.fx + cam_a.fy + cam_b.fx + cam_b.fy)
                E, mask = cv2.findEssentialMat(xa, xb, np.eye(3), method=cv2.RANSAC,  # noqa: N806
                                               prob=0.999,
                                               threshold=opt.relative_pose_threshold_px / f)
                if E is None or E.shape[0] < 3:
                    continue
                _, R, t, mask = cv2.recoverPose(E[:3], xa, xb, np.eye(3), mask=mask)  # noqa: N806
                inliers = mask.ravel() > 0
                t = t.ravel() / np.linalg.norm(t)
            else:
                raise ValueError(f"unknown relative_pose_solver {opt.relative_pose_solver}")
            if inliers.sum() < opt.min_pair_inliers:
                continue
            P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # noqa: N806
            P2 = np.hstack([R, t.reshape(3, 1)])  # noqa: N806
            X = triangulate_dlt(P1, P2, xa[inliers], xb[inliers])  # noqa: N806
            angles = triangulation_angles_deg(np.zeros(3), -R.T @ t, X)
            self.pairs[(a, b)] = PairGeometry(m[inliers], R, t.ravel(),
                                              float(np.median(angles)))

    def choose_initial_pair(self, exclude: set | None = None) -> tuple[int, int] | None:
        """Most inliers among pairs with a wide enough median triangulation angle.

        The pair with the most matches is usually the one with the *shortest* baseline
        (most overlap), whose depth is barely constrained. Requiring a minimum median
        angle trades a few inliers for a well-conditioned first model; if no pair
        qualifies, fall back to inliers x angle.
        """
        exclude = exclude or set()
        cands = [(p, g) for p, g in self.pairs.items() if p not in exclude]
        if not cands:
            return None
        min_angle = self.options.init_min_triangulation_angle_deg
        wide = [(p, g) for p, g in cands if g.median_angle_deg >= min_angle]
        if wide:
            return max(wide, key=lambda pg: (len(pg[1].matches), pg[0]))[0]
        return max(cands, key=lambda pg: len(pg[1].matches) * pg[1].median_angle_deg)[0]

    def next_image(self, exclude: set[int] | None = None) -> int | None:
        """Unregistered image with the most 2D-3D correspondences (ties -> lowest id)."""
        exclude = exclude or set()
        has_point = np.zeros(len(self.tracks) + 1, dtype=bool)
        for tid in self.track_point:
            has_point[tid] = True
        best, best_count = None, self.options.min_pnp_inliers - 1
        for iid in sorted(self.recon.images):
            if iid in exclude or self.recon.images[iid].is_registered:
                continue
            tids = self.kp_track[iid]
            count = int(has_point[tids[tids >= 0]].sum())
            if count > best_count:
                best, best_count = iid, count
        return best

    def register_image(self, image_id: int) -> bool:
        im = self.recon.images[image_id]
        tids = self.kp_track[image_id]
        kps = np.array([k for k, tid in enumerate(tids) if tid >= 0 and tid in self.track_point],
                       dtype=np.int64)
        if len(kps) < self.options.min_pnp_inliers:
            return False
        pids = [self.track_point[int(tids[k])] for k in kps]
        X = np.stack([self.recon.points[p].xyz for p in pids])  # noqa: N806
        camera = self.cameras[im.camera_id]
        pose, inliers = pnp_ransac(X, im.keypoints[kps], camera,
                                   threshold_px=self.options.pnp_threshold_px, rng=self.rng)
        if pose is None or inliers.sum() < self.options.min_pnp_inliers:
            return False
        im.pose = pose
        for k, pid, ok in zip(kps, pids, inliers, strict=True):
            if ok:
                self.recon.points[pid].track.append((image_id, int(k)))
                im.point3d_ids[k] = pid
        return True

    def triangulate_image(self, image_id: int) -> int:
        """Triangulate every track seen in `image_id` that has >= 2 registered observations
        and no 3D point yet. Returns the number of new points."""
        created = 0
        for tid in self.kp_track[image_id]:
            if tid < 0 or tid in self.track_point:
                continue
            obs = [(i, kp) for i, kp in self.tracks[tid] if self.recon.images[i].is_registered]
            if len(obs) >= 2 and self._triangulate_track(int(tid), obs):
                created += 1
        return created

    def filter_points(self) -> int:
        """Drop observations with large reprojection error or negative depth, then points
        with < 2 observations or too small a triangulation angle. Returns #points removed."""
        opt = self.options
        recon = self.recon
        removed = 0
        for pid in list(recon.points):
            pt = recon.points[pid]
            keep = []
            for image_id, kp in pt.track:
                im = recon.images[image_id]
                uv, z = project(self.cameras[im.camera_id], im.pose, pt.xyz[None])
                err = np.linalg.norm(uv[0] - im.keypoints[kp])
                if z[0] > 0 and err <= opt.max_reprojection_error_px:
                    keep.append((image_id, kp))
                else:
                    im.point3d_ids[kp] = -1
            pt.track = keep
            if len(keep) < 2 or self._max_angle_deg(pt.xyz, keep) < opt.min_triangulation_angle_deg:
                self._delete_point(pid)
                removed += 1
        return removed

    # Internals --------------------------------------------------------------------------

    def _fresh_images(self) -> dict[int, Image]:
        return {iid: Image(iid, im.camera_id, im.name, keypoints=im.keypoints.copy())
                for iid, im in self.images.items()}

    def _reconstruct(self, pair: tuple[int, int]) -> Reconstruction:
        opt = self.options
        a, b = pair
        geom = self.pairs[pair]
        self.recon = Reconstruction(cameras=dict(self.cameras), images=self._fresh_images())
        self.track_point: dict[int, int] = {}
        self.point_track: dict[int, int] = {}
        self.recon.images[a].pose = Pose.identity()
        self.recon.images[b].pose = Pose(geom.R, geom.t)
        self.fixed = {a}
        self.triangulate_image(b)
        self._global_ba(max_iterations=50)
        if len(self.recon.points) < opt.min_pnp_inliers:
            return self.recon

        last_ba = 2
        failed: set[int] = set()
        while True:
            nxt = self.next_image(exclude=failed)
            if nxt is None:
                break
            if not self.register_image(nxt):
                failed.add(nxt)
                continue
            failed.clear()  # new structure: previously failed images may now succeed
            new_points = self.triangulate_image(nxt)
            n_reg = len(self.recon.registered_image_ids)
            self._log(f"  registered {self.recon.images[nxt].name} ({n_reg}/{len(self.images)})"
                      f", +{new_points} points")
            if n_reg <= 6 or n_reg - last_ba >= opt.global_ba_every:
                self._global_ba(max_iterations=50)
                last_ba = n_reg

        # Final: retriangulate with refined poses, then polish.
        for iid in self.recon.registered_image_ids:
            self.triangulate_image(iid)
        self._global_ba(max_iterations=100)
        self._global_ba(max_iterations=100)
        return self.recon

    def _global_ba(self, max_iterations: int) -> None:
        report = bundle_adjust(self.recon, fixed_image_ids=self.fixed,
                               max_iterations=max_iterations)
        removed = self.filter_points()
        self._log(f"  BA: rmse {report.initial_rmse_px:.3f} -> {report.final_rmse_px:.3f} px, "
                  f"{len(self.recon.points)} points (-{removed})")

    def _triangulate_track(self, tid: int, obs: list[tuple[int, int]]) -> bool:
        opt = self.options
        recon = self.recon
        for _ in range(2):
            Ps = [recon.images[i].pose.matrix()[:3] for i, _ in obs]  # noqa: N806
            xs = np.stack([self.xn[i][kp] for i, kp in obs])
            A = []  # noqa: N806
            for P, (u, v) in zip(Ps, xs, strict=True):  # noqa: N806
                A.append(u * P[2] - P[0])
                A.append(v * P[2] - P[1])
            _, _, Vt = np.linalg.svd(np.asarray(A))  # noqa: N806
            if abs(Vt[-1, 3]) < 1e-12:
                return False
            X = Vt[-1, :3] / Vt[-1, 3]  # noqa: N806
            keep = []
            for i, kp in obs:
                im = recon.images[i]
                uv, z = project(self.cameras[im.camera_id], im.pose, X[None])
                if z[0] > 0 and np.linalg.norm(uv[0] - im.keypoints[kp]) <= \
                        opt.max_reprojection_error_px:
                    keep.append((i, kp))
            if len(keep) == len(obs):
                break
            if len(keep) < 2:
                return False
            obs = keep
        else:
            return False
        if self._max_angle_deg(X, obs) < opt.min_triangulation_angle_deg:
            return False
        pid = recon.add_point(X, obs)
        self.track_point[tid] = pid
        self.point_track[pid] = tid
        return True

    def _max_angle_deg(self, X: np.ndarray, obs: list[tuple[int, int]]) -> float:  # noqa: N803
        centers = np.stack([self.recon.images[i].pose.center for i, _ in obs])
        rays = centers - X
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        cos = np.clip(rays @ rays.T, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos.min())))

    def _delete_point(self, pid: int) -> None:
        pt = self.recon.points.pop(pid)
        for image_id, kp in pt.track:
            self.recon.images[image_id].point3d_ids[kp] = -1
        tid = self.point_track.pop(pid, None)
        if tid is not None:
            self.track_point.pop(tid, None)
