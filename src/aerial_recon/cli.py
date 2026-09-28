"""Command-line entry point: `uv run aerial-recon <command>`."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

MILESTONES = {
    1: "Camera geometry & SO(3)",
    2: "Video -> keyframes",
    3: "Features, matching, RANSAC, homography",
    4: "Two-view geometry & triangulation",
    5: "Bundle adjustment",
    6: "Incremental SfM",
    7: "Georeferencing & pose evaluation",
    8: "Plane-sweep MVS",
    9: "Fusion & geometry evaluation",
}


def _status(args: argparse.Namespace) -> int:
    import pytest

    class Collector:
        def __init__(self) -> None:
            self.results: dict[int, list[bool]] = defaultdict(list)

        def pytest_runtest_logreport(self, report) -> None:
            if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
                m = re.search(r"test_m(\d+)_", report.nodeid)
                if m:
                    self.results[int(m.group(1))].append(report.outcome == "passed")

    collector = Collector()
    tests = Path(__file__).resolve().parents[2] / "tests" / "student"
    pytest_args = [str(tests), "-q", "-p", "no:cacheprovider", "--no-header", "-o", "addopts=",
                   "--tb=no"]
    if args.fast:
        pytest_args += ["-m", "not slow"]
    pytest.main(pytest_args, plugins=[collector])
    print("\nMilestone progress (tests/student)")
    print(f"{'':3} {'milestone':44} {'passed':>8}")
    next_up = None
    for num, title in MILESTONES.items():
        res = collector.results.get(num, [])
        done = bool(res) and all(res)
        if not done and next_up is None:
            next_up = num
        mark = "✅" if done else ("🟡" if any(res) else "⬜")
        print(f"{mark:3} M{num:<2} {title:40} {sum(res):>3}/{len(res):<3}")
    print("⬜  M10+ modern track: gates are scripted experiments, see docs/roadmap.md")
    if next_up is not None:
        print(f"\nNext: M{next_up} — uv run pytest tests/student -k m{next_up:02d} -x")
    return 0


def _download(args: argparse.Namespace) -> int:
    from aerial_recon.datasets import DATASETS, download

    if args.name == "list":
        for d in DATASETS.values():
            print(f"{d.name:16} {d.images:4} imgs {d.size:>8}  {d.license:14} {d.notes}")
        return 0
    download(args.name)
    return 0


def _extract_frames(args: argparse.Namespace) -> int:
    from aerial_recon.video.frames import extract_frames

    paths = extract_frames(args.video, args.out, every_n=args.every_n,
                           max_frames=args.max_frames, resize_width=args.width)
    print(f"wrote {len(paths)} frames to {args.out}")
    return 0


def _colmap(args: argparse.Namespace) -> int:
    from aerial_recon.sfm.reference import run_colmap

    recon = run_colmap(args.images, args.out, matcher=args.matcher, mapper=args.mapper,
                       max_image_size=args.max_size, overwrite=args.overwrite)
    print(recon.summary())
    return 0


def _sfm(args: argparse.Namespace) -> int:
    """Your pipeline end to end on real images (M6 gate)."""
    import cv2
    import numpy as np

    from aerial_recon.io.colmap_text import write_model
    from aerial_recon.modern.pose_sources import load_poses
    from aerial_recon.sfm.features import detect_sift, match_descriptors, root_sift
    from aerial_recon.sfm.incremental import IncrementalSfM, SfMOptions
    from aerial_recon.types import Image

    ref = load_poses(args.intrinsics)
    camera = next(iter(ref.cameras.values()))
    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    images, descriptors = {}, {}
    cache = args.cache
    cached = None
    if cache is not None and cache.exists():
        cached = np.load(cache, allow_pickle=False)
        if [str(n) for n in cached["names"]] != [p.name for p in paths]:
            print(f"cache {cache} is for a different image set; ignoring it")
            cached = None
    if cached is not None:
        for image_id, path in enumerate(paths, start=1):
            images[image_id] = Image(image_id, 1, path.name,
                                     keypoints=cached[f"kp_{image_id}"])
        print(f"loaded keypoints for {len(paths)} images from {cache}")
    else:
        for image_id, path in enumerate(paths, start=1):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            factor = min(1.0, args.max_size / max(gray.shape))
            if factor < 1.0:
                gray = cv2.resize(gray, None, fx=factor, fy=factor,
                                  interpolation=cv2.INTER_AREA)
            kps, desc = detect_sift(gray, max_features=args.max_features)
            images[image_id] = Image(image_id, 1, path.name, keypoints=kps / factor)
            descriptors[image_id] = root_sift(desc)
            print(f"[{image_id}/{len(paths)}] {path.name}: {len(kps)} features", flush=True)

    ids = sorted(images)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
             if args.sequential <= 0 or b - a <= args.sequential]
    matches = {}
    if cached is not None:
        for a, b in pairs:
            key = f"m_{a}_{b}"
            if key in cached:
                matches[(a, b)] = cached[key]
    else:
        for n, (a, b) in enumerate(pairs, start=1):
            m = match_descriptors(descriptors[a], descriptors[b])
            if len(m) >= 15:
                matches[(a, b)] = m
            if n % 100 == 0:
                print(f"  matched {n}/{len(pairs)} pairs", flush=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, names=np.array([p.name for p in paths]),
                                **{f"kp_{i}": images[i].keypoints for i in ids},
                                **{f"m_{a}_{b}": m for (a, b), m in matches.items()})
    print(f"{len(matches)} / {len(pairs)} pairs with >= 15 matches")
    solver = "five_point_opencv" if args.five_point else "eight_point"
    recon = IncrementalSfM({1: camera}, images, matches,
                           SfMOptions(relative_pose_solver=solver, verbose=True)).run()
    write_model(recon, args.out)
    print(recon.summary())
    print(f"mean inlier matches per pair: {np.mean([len(m) for m in matches.values()]):.0f}")
    return 0


def _compare_poses(args: argparse.Namespace) -> int:
    import numpy as np

    from aerial_recon.eval.poses import absolute_trajectory_error, pose_auc, relative_pose_errors
    from aerial_recon.modern.pose_sources import load_poses

    est, ref = load_poses(args.estimate), load_poses(args.reference)
    est_by_name = {im.name: im for im in est.images.values() if im.is_registered}
    ref_by_name = {im.name: im for im in ref.images.values() if im.is_registered}
    common = sorted(set(est_by_name) & set(ref_by_name))
    print(f"registered: estimate {len(est_by_name)}, reference {len(ref_by_name)}, "
          f"common {len(common)}")
    ate = absolute_trajectory_error(
        np.stack([est_by_name[n].pose.center for n in common]),
        np.stack([ref_by_name[n].pose.center for n in common]))
    print(f"ATE (Sim3-aligned, reference units): rmse {ate.rmse:.4f}  median {ate.median:.4f}")
    rot, trans = relative_pose_errors(est, ref)
    thresholds = [3.0, 5.0, 10.0, 30.0]
    aucs = pose_auc(np.maximum(rot, trans), thresholds)
    print("pairwise AUC " + "  ".join(f"@{t:g}°={a:.3f}" for t, a in zip(thresholds, aucs,
                                                                          strict=True)))
    print(f"median rotation error {np.median(rot):.3f}°, translation {np.median(trans):.3f}°")
    if args.json:
        import json

        payload = {
            "registered_estimate": len(est_by_name), "registered_reference": len(ref_by_name),
            "common": len(common), "ate_rmse": ate.rmse, "ate_median": ate.median,
            "ate_max": ate.max, "sim3_scale": ate.scale,
            "auc": {f"{t:g}": a for t, a in zip(thresholds, aucs, strict=True)},
            "median_rotation_error_deg": float(np.median(rot)),
            "median_translation_error_deg": float(np.median(trans)),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
    return 0


def _georef(args: argparse.Namespace) -> int:
    """M7 gate: EXIF GPS -> ENU -> Sim3 from camera centers."""
    from aerial_recon.geo.georef import georeference, write_georef
    from aerial_recon.io.colmap_text import write_model
    from aerial_recon.modern.pose_sources import load_poses

    recon = load_poses(args.model)
    result = georeference(recon, args.images)
    write_model(recon, args.out)
    summary = write_georef(result, args.out / "georef.json")
    print(f"georeferenced {summary['num_images']} images, scale {summary['scale']:.4f}")
    print(f"GPS residuals: rmse 3D {summary['rmse_3d_m']:.2f} m, horizontal "
          f"{summary['rmse_horizontal_m']:.2f} m, vertical {summary['rmse_vertical_m']:.2f} m, "
          f"max {summary['max_3d_m']:.2f} m")
    slope = summary.get("vertical_residual_vs_radius_slope")
    if slope is not None:
        print(f"vertical residual vs radius slope {slope:+.4f} m/m (doming check), planar "
              f"trend R² (E, N, U) = "
              + ", ".join(f"{v:.2f}" for v in summary["residual_planar_trend_r2_enu"]))
    return 0


def _mvs(args: argparse.Namespace) -> int:
    """M8-M9 deliverable: model + images -> depth maps -> fused PLY (+ Poisson mesh)."""
    from aerial_recon.modern.pose_sources import load_poses
    from aerial_recon.mvs.pipeline import MVSOptions, run_mvs

    recon = load_poses(args.model)
    opt = MVSOptions(max_image_size=args.max_size, num_sources=args.sources,
                     depth_hypotheses=args.hypotheses, window=args.window,
                     min_score=args.min_score, min_consistent=args.min_consistent,
                     voxel_size=args.voxel, workers=args.workers)
    summary = run_mvs(recon, args.images, args.out, opt)
    print(f"{summary['num_points']} fused points from {summary['num_views']} views in "
          f"{summary['seconds']:.0f}s -> {args.out / 'fused.ply'}")
    if args.mesh:
        try:
            import numpy as np

            from aerial_recon.mvs.meshing import poisson_mesh, write_mesh
        except ImportError:
            print("meshing needs open3d: uv sync --extra mesh", file=sys.stderr)
            return 1
        data = np.load(args.out / "fused.npz")
        mesh = poisson_mesh(data["points"], data["colors"], depth=args.poisson_depth)
        write_mesh(args.out / "mesh.ply", mesh)
        print(f"mesh -> {args.out / 'mesh.ply'}")
    return 0


def _load_points(path: Path):
    import numpy as np

    from aerial_recon.io.ply import read_ply_points

    if path.suffix == ".npz":
        return np.load(path)["points"].astype(np.float64)
    return read_ply_points(path)


def _eval_geometry(args: argparse.Namespace) -> int:
    """M9 gate: precision / recall / F-score of a fused cloud against a reference."""
    import json

    from aerial_recon.eval.reference import evaluate_against_reference, load_laz_enu
    from aerial_recon.geo.georef import read_ref_lla

    pred = _load_points(args.prediction)
    if args.reference.suffix.lower() in {".laz", ".las"}:
        if args.georef is None:
            print("--georef georef.json is required to put a LAS/LAZ reference in ENU",
                  file=sys.stderr)
            return 2
        reference, _ = load_laz_enu(args.reference, read_ref_lla(args.georef))
    else:
        reference = _load_points(args.reference)
    metrics = evaluate_against_reference(pred, reference, args.thresholds,
                                         use_icp=not args.no_icp)
    print(f"prediction {metrics['num_pred']} points ({metrics['num_pred_in_footprint']} in "
          f"reference footprint), reference {metrics['num_reference']} points")
    if "icp" in metrics:
        icp = metrics["icp"]
        print("ICP refinement of GPS alignment: translation "
              + " ".join(f"{v:+.2f}" for v in icp["translation_m"])
              + f" m, rotation {icp['rotation_deg']:.3f}°")
    for key in ("gps_aligned", "icp_aligned"):
        if key not in metrics:
            continue
        print(f"{key}:")
        for tau, m in metrics[key].items():
            print(f"  τ={float(tau) * 100:>4.0f} cm  P {m['precision']:.3f}  R {m['recall']:.3f}  "
                  f"F {m['fscore']:.3f}  acc {m['accuracy']:.3f} m  "
                  f"comp {m['completeness']:.3f} m")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aerial-recon")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="milestone progress from the student tests")
    p.add_argument("--fast", action="store_true", help="skip tests marked slow")
    p.set_defaults(handler=_status)

    p = sub.add_parser("download", help="fetch a small public drone dataset into data/")
    p.add_argument("name", help="dataset name, or 'list'")
    p.set_defaults(handler=_download)

    p = sub.add_parser("extract-frames", help="decode a video into PNG frames")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every-n", type=int, default=5)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--width", type=int, help="resize frames to this width")
    p.set_defaults(handler=_extract_frames)

    p = sub.add_parser("colmap", help="reference reconstruction with COLMAP / GLOMAP")
    p.add_argument("images", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    p.add_argument("--mapper", choices=["incremental", "global"], default="incremental")
    p.add_argument("--max-size", type=int, default=2000)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(handler=_colmap)

    p = sub.add_parser("sfm", help="run YOUR incremental SfM on real images (M6 gate)")
    p.add_argument("images", type=Path)
    p.add_argument("--intrinsics", type=Path, required=True,
                   help="COLMAP model whose first camera provides intrinsics")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-size", type=int, default=1600)
    p.add_argument("--max-features", type=int, default=8000)
    p.add_argument("--five-point", action="store_true",
                   help="use OpenCV's 5-point essential solver for pair verification")
    p.add_argument("--sequential", type=int, default=0,
                   help="only match images within this index distance (0 = exhaustive)")
    p.add_argument("--cache", type=Path,
                   help="npz cache of keypoints + raw matches (created if missing)")
    p.set_defaults(handler=_sfm)

    p = sub.add_parser("compare-poses", help="evaluate a COLMAP-format model against a reference")
    p.add_argument("estimate", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("--json", type=Path, help="also write the metrics to this JSON file")
    p.set_defaults(handler=_compare_poses)

    p = sub.add_parser("georef", help="GPS-align a model into local ENU metres (M7 gate)")
    p.add_argument("model", type=Path, help="COLMAP-format model (text or binary)")
    p.add_argument("images", type=Path, help="image directory with EXIF GPS")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=_georef)

    p = sub.add_parser("mvs", help="plane-sweep MVS + fusion: model + images -> fused PLY")
    p.add_argument("model", type=Path)
    p.add_argument("images", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-size", type=int, default=1600)
    p.add_argument("--sources", type=int, default=4)
    p.add_argument("--hypotheses", type=int, default=128)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--min-score", type=float, default=0.5, help="min winning mean ZNCC")
    p.add_argument("--min-consistent", type=int, default=2)
    p.add_argument("--voxel", type=float, default=0.05, help="fusion voxel size (model units)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mesh", action="store_true", help="Poisson mesh (needs --extra mesh)")
    p.add_argument("--poisson-depth", type=int, default=10)
    p.set_defaults(handler=_mvs)

    p = sub.add_parser("eval-geometry", help="P/R/F-score of a fused cloud vs a reference")
    p.add_argument("prediction", type=Path, help="fused.npz or .ply (ENU)")
    p.add_argument("--reference", type=Path, required=True, help=".laz/.las or ENU .ply/.npz")
    p.add_argument("--georef", type=Path, help="georef.json (needed for LAS/LAZ references)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--no-icp", action="store_true", help="skip ICP refinement")
    p.add_argument("--out", type=Path, help="write metrics JSON here")
    p.set_defaults(handler=_eval_geometry)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
