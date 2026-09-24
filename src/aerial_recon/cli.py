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
    for image_id, path in enumerate(paths, start=1):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        factor = min(1.0, args.max_size / max(gray.shape))
        if factor < 1.0:
            gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        kps, desc = detect_sift(gray, max_features=args.max_features)
        images[image_id] = Image(image_id, 1, path.name, keypoints=kps / factor)
        descriptors[image_id] = root_sift(desc)
        print(f"[{image_id}/{len(paths)}] {path.name}: {len(kps)} features", flush=True)

    ids = sorted(images)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
             if args.sequential <= 0 or b - a <= args.sequential]
    matches = {}
    for a, b in pairs:
        m = match_descriptors(descriptors[a], descriptors[b])
        if len(m) >= 15:
            matches[(a, b)] = m
    print(f"{len(matches)} / {len(pairs)} pairs with >= 15 matches")
    solver = "five_point_opencv" if args.five_point else "eight_point"
    recon = IncrementalSfM({1: camera}, images, matches,
                           SfMOptions(relative_pose_solver=solver)).run()
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
    p.set_defaults(handler=_sfm)

    p = sub.add_parser("compare-poses", help="evaluate a COLMAP-format model against a reference")
    p.add_argument("estimate", type=Path)
    p.add_argument("reference", type=Path)
    p.set_defaults(handler=_compare_poses)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
