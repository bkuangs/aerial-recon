"""M8-M9 real-data driver: sparse model + images -> depth maps -> fused point cloud (+ mesh).

    undistort + downscale  ->  pick source views  ->  plane sweep per view
    ->  confidence filter  ->  geometric-consistency fusion  ->  voxel average  ->  PLY

Run it on a georeferenced model (`aerial-recon georef`) so the output is metric ENU and
voxel sizes / thresholds are in metres.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from aerial_recon.geometry.camera import distort
from aerial_recon.io.ply import write_ply
from aerial_recon.mvs.fusion import fuse_depth_maps
from aerial_recon.mvs.plane_sweep import plane_sweep_depth
from aerial_recon.types import Camera, Reconstruction


@dataclass
class MVSOptions:
    max_image_size: int = 1600
    num_sources: int = 4
    depth_hypotheses: int = 128
    window: int = 7
    min_score: float = 0.5
    min_consistent: int = 2
    fusion_neighbors: int = 8
    max_reprojection_px: float = 1.0
    max_relative_depth: float = 0.01
    voxel_size: float = 0.05
    workers: int = 4


def undistort_image(image: np.ndarray, camera: Camera) -> tuple[np.ndarray, Camera]:
    """Resample `image` (already at `camera`'s resolution) to a distortion-free pinhole
    camera with the same K. Uses the COLMAP pixel convention (centers at +0.5)."""
    pinhole = Camera(camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy)
    if not camera.has_distortion:
        return image, pinhole
    cols, rows = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    xy = np.column_stack([((cols + 0.5 - camera.cx) / camera.fx).ravel(),
                          ((rows + 0.5 - camera.cy) / camera.fy).ravel()])
    xy_d = distort(camera, xy)
    map_x = (xy_d[:, 0] * camera.fx + camera.cx - 0.5).reshape(rows.shape).astype(np.float32)
    map_y = (xy_d[:, 1] * camera.fy + camera.cy - 0.5).reshape(rows.shape).astype(np.float32)
    out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out, pinhole


def load_view(path: Path, camera: Camera, max_size: int) -> tuple[np.ndarray, np.ndarray, Camera]:
    """(gray float32 [0,1], rgb uint8, undistorted scaled camera)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    factor = min(1.0, max_size / max(camera.width, camera.height))
    cam = camera.scaled(factor)
    if (bgr.shape[1], bgr.shape[0]) != (cam.width, cam.height):
        bgr = cv2.resize(bgr, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    bgr, cam = undistort_image(bgr, cam)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray, rgb, cam


def view_neighbors(recon: Reconstruction, image_ids: list[int], k: int,
                   min_angle_deg: float = 1.0) -> dict[int, list[int]]:
    """Rank other views by shared sparse points, weighted toward useful baselines.

    score = shared points x min(1, median triangulation angle / 5°); pairs whose median
    angle is below `min_angle_deg` are skipped (too little parallax for stereo).
    """
    obs: dict[int, set[int]] = {i: set() for i in image_ids}
    for pid, pt in recon.points.items():
        for iid, _ in pt.track:
            if iid in obs:
                obs[iid].add(pid)
    centers = {i: recon.images[i].pose.center for i in image_ids}
    out = {}
    for i in image_ids:
        scored = []
        for j in image_ids:
            if j == i:
                continue
            shared = list(obs[i] & obs[j])
            if len(shared) < 10:
                continue
            X = np.stack([recon.points[p].xyz for p in shared])  # noqa: N806
            r1 = centers[i] - X
            r2 = centers[j] - X
            cos = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
            angle = float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))))
            if angle < min_angle_deg:
                continue
            scored.append((len(shared) * min(1.0, angle / 5.0), j))
        out[i] = [j for _, j in sorted(scored, reverse=True)[:k]]
    return out


def depth_range(recon: Reconstruction, image_id: int) -> tuple[float, float] | None:
    im = recon.images[image_id]
    pids = [p for p in im.point3d_ids if p >= 0 and p in recon.points]
    if len(pids) < 10:
        return None
    X = np.stack([recon.points[p].xyz for p in pids])  # noqa: N806
    z = im.pose.transform(X)[:, 2]
    z = z[z > 0]
    lo, hi = np.percentile(z, [1, 99])
    return 0.8 * lo, 1.25 * hi


def _sweep_job(args):
    ref, ref_cam, ref_pose, srcs, src_cams, src_poses, depths, window = args
    return plane_sweep_depth(ref, ref_cam, ref_pose, srcs, src_cams, src_poses, depths, window)


def run_mvs(recon: Reconstruction, image_dir: str | Path, out_dir: str | Path,
            options: MVSOptions | None = None, log=None) -> dict:
    opt = options or MVSOptions()
    if log is None:
        def log(msg: str) -> None:
            print(msg, flush=True)
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids = recon.registered_image_ids
    grays, rgbs, cams, poses = {}, {}, {}, {}
    for iid in ids:
        im = recon.images[iid]
        grays[iid], rgbs[iid], cams[iid] = load_view(image_dir / im.name,
                                                     recon.cameras[im.camera_id],
                                                     opt.max_image_size)
        poses[iid] = im.pose
    cam0 = cams[ids[0]]
    log(f"loaded {len(ids)} views at {cam0.width}x{cam0.height} ({time.time() - t0:.0f}s)")

    k = max(opt.num_sources, opt.fusion_neighbors)
    neighbors = view_neighbors(recon, ids, k)
    jobs, job_ids, ranges = [], [], {}
    for iid in ids:
        srcs = neighbors[iid][:opt.num_sources]
        rng = depth_range(recon, iid)
        if not srcs or rng is None:
            log(f"  skip {recon.images[iid].name}: no sources or sparse depth")
            continue
        ranges[iid] = rng
        # Uniform in inverse depth: equal pixel-disparity steps for any baseline.
        depths = 1.0 / np.linspace(1.0 / rng[0], 1.0 / rng[1], opt.depth_hypotheses)
        jobs.append((grays[iid], cams[iid], poses[iid], [grays[j] for j in srcs],
                     [cams[j] for j in srcs], [poses[j] for j in srcs], depths, opt.window))
        job_ids.append(iid)

    depth_maps: dict[int, np.ndarray] = {}
    stats = {}
    for iid, (depth, score) in zip(job_ids, _bounded_map(_sweep_job, jobs, opt.workers),
                                   strict=True):
        lo, hi = ranges[iid]
        keep = (score > opt.min_score) & (depth > lo * 1.001) & (depth < hi * 0.999)
        filtered = np.where(keep, depth, 0.0).astype(np.float32)
        depth_maps[iid] = filtered
        np.savez_compressed(out_dir / "depth" / f"{Path(recon.images[iid].name).stem}.npz",
                            depth=depth, score=score)
        stats[recon.images[iid].name] = {"valid_fraction": float(keep.mean()),
                                         "median_score": float(np.median(score))}
        log(f"  depth {recon.images[iid].name}: {keep.mean():.1%} valid "
            f"({time.time() - t0:.0f}s)")

    order = [i for i in ids if i in depth_maps]
    index = {iid: n for n, iid in enumerate(order)}
    fusion_nb = [[index[j] for j in neighbors[i][:opt.fusion_neighbors] if j in index]
                 for i in order]
    points, colors = fuse_depth_maps(
        [depth_maps[i] for i in order], [cams[i] for i in order], [poses[i] for i in order],
        colors=[rgbs[i] for i in order], min_consistent=opt.min_consistent,
        max_reprojection_px=opt.max_reprojection_px, max_relative_depth=opt.max_relative_depth,
        neighbors=fusion_nb, voxel_size=opt.voxel_size)
    log(f"fused {len(points)} points ({time.time() - t0:.0f}s)")
    np.savez_compressed(out_dir / "fused.npz", points=points.astype(np.float32),
                        colors=np.clip(colors, 0, 255).astype(np.uint8))
    write_ply(out_dir / "fused.ply", points, np.clip(colors, 0, 255).astype(np.uint8))
    summary = {"options": asdict(opt), "num_views": len(order), "num_points": int(len(points)),
               "resolution": [cam0.width, cam0.height], "seconds": time.time() - t0,
               "views": stats}
    (out_dir / "mvs.json").write_text(json.dumps(summary, indent=2))
    return summary


def _bounded_map(fn, jobs, workers: int):
    """Ordered parallel map that keeps at most 2 x workers jobs in flight (RAM bound)."""
    if workers <= 1:
        yield from (fn(j) for j in jobs)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        it = iter(jobs)
        for job in it:
            pending.append(pool.submit(fn, job))
            if len(pending) >= 2 * workers:
                yield pending.pop(0).result()
        for fut in pending:
            yield fut.result()
