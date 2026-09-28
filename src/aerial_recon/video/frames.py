"""M2 — From video to a good image set.

Video is a terrible SfM input if used naively: 30 fps gives thousands of near-duplicate,
motion-blurred, rolling-shutter frames with tiny baselines. Choosing frames that are
sharp *and* have enough parallax between them is the first real engineering decision.

The scaffold functions (extract_frames, read_gray) are provided; the scoring and selection
functions are yours.

Reading: Pertuz et al. "Analysis of focus measure operators for shape-from-focus" (2013);
Szeliski 2e §9.1 (optical flow); Lucas & Kanade (1981).
"""

from __future__ import annotations

from pathlib import Path
from scipy import ndimage

import cv2
import numpy as np


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    every_n: int = 1,
    max_frames: int | None = None,
    resize_width: int | None = None,
) -> list[Path]:
    """Decode a video and write every `every_n`-th frame as PNG. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    paths: list[Path] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % every_n == 0:
            if resize_width is not None and frame.shape[1] != resize_width:
                h = round(frame.shape[0] * resize_width / frame.shape[1])
                frame = cv2.resize(frame, (resize_width, h), interpolation=cv2.INTER_AREA)
            path = out_dir / f"frame_{index:06d}.png"
            cv2.imwrite(str(path), frame)
            paths.append(path)
            if max_frames is not None and len(paths) >= max_frames:
                break
        index += 1
    cap.release()
    return paths


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as float32 grayscale in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def sharpness(gray: np.ndarray) -> float:
    """
    Determine the sharpness of an image viahe Laplacian kernel.

    Higher = sharper. Must be invariant to adding a constant to the image.
    """
    gray = gray.astype(np.float64)

    L = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    return float(np.var(ndimage.convolve(gray, L)))


def frame_displacement(gray_a: np.ndarray, gray_b: np.ndarray, max_corners: int = 400) -> float:
    """
    Determine the image-space motion (pixels) of tracked corners from frame a to frame b.

    Suggested: cv2.goodFeaturesToTrack on a, cv2.calcOpticalFlowPyrLK to b, keep points with
    status == 1, return the median displacement magnitude. This is a cheap parallax proxy.
    Inputs are float [0, 1] grayscale; convert to uint8 for OpenCV.
    """
    a8 = (gray_a * 255).astype(np.uint8)
    b8 = (gray_b * 255).astype(np.uint8)

    pts = cv2.goodFeaturesToTrack(      # Shi-Tomashi corner detection
        a8, 
        maxCorners=max_corners, 
        qualityLevel=0.01, 
        minDistance=7
        )

    if pts is None: 
        return 0.0

    # Lucas-Kanade optical flow with coarse-to-fine
    # nxt: new position of tracked corners in frame b
    # status: bool mask tells us whethe a corner was detected in frame b
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a8, b8, pts, None)

    ok = status.ravel() == 1    # flatten to 1D
    if not ok.any():
        return 0.0

    # nxt - pts = motion vector
    return float(np.median(np.linalg.norm(nxt[ok] - pts[ok], axis=-1))) # convert vector to pixel distance

def select_keyframes(
    sharpness_scores: np.ndarray,
    displacements: np.ndarray,
    min_displacement: float,
    search_window: int = 5,
) -> list[int]:
    """
    Choose frames that are "sharp" and spaced by at least `min_displacement` pixels to
    use for SfM.

    Args:
        sharpness_scores: (N,) per-frame sharpness.
        displacements: (N-1,) motion between consecutive frames i -> i+1.
        min_displacement: required cumulative motion since the previous keyframe.
        search_window: once the motion threshold is first reached at frame j, pick the
            sharpest frame among j .. j+search_window-1 (clipped to N).

    Algorithm (implement exactly so the tests are deterministic):
        1. First keyframe = sharpest among frames 0 .. search_window-1 (ties -> earliest).
        2. From the last keyframe k, find the smallest j > k with
           sum(displacements[k:j]) >= min_displacement. If none, stop.
        3. Append the sharpest frame in [j, min(j + search_window, N)) (ties -> earliest).
        4. Repeat from step 2.
    """
    n = len(sharpness_scores)

    # This is the reference image A. Pick the best one
    keyframes = [int(np.argmax(sharpness_scores[:search_window]))]   # step 1

    while True:
        k = keyframes[-1]
        total = 0.0
        j = None
        # Choose the first image that has moved enough
        for m in range(k + 1, n):                   # step 2: smallest j with enough motion
            total += displacements[m - 1]
            if total >= min_displacement:
                j = m
                break
        if j is None:
            break
        # But it may be blurry. Find the sharpest
        hi = min(j + search_window, n)              # step 3: sharpest in [j, hi)
        keyframes.append(j + int(np.argmax(sharpness_scores[j:hi])))

    return keyframes


def extract_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    target: int | None = 150,
    min_displacement: float | None = None,
    analysis_width: int = 640,
    resize_width: int | None = 1920,
    search_window: int = 9,
    stride: int = 1,
    jpeg_quality: int = 95,
    log=print,
) -> tuple[list[Path], dict]:
    """Video -> sharp keyframes with roughly uniform parallax.

    Pass 1 decodes every `stride`-th frame at `analysis_width`, scoring sharpness and the
    LK median displacement to the previous analysed frame. `select_keyframes` then picks
    the frames; `min_displacement` (px at analysis width) defaults to total motion / target.
    Pass 2 re-decodes and writes only the selected frames as JPEG. Returns (paths, stats).
    """
    video_path, out_dir = Path(video_path), Path(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_ids, scores, disps = [], [], []
    prev = None
    index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % stride == 0:
            _, frame = cap.retrieve()
            h = round(frame.shape[0] * analysis_width / frame.shape[1])
            small = cv2.resize(frame, (analysis_width, h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            scores.append(sharpness(gray))
            if prev is not None:
                disps.append(frame_displacement(prev, gray))
            prev = gray
            frame_ids.append(index)
            if len(frame_ids) % 1000 == 0:
                log(f"  analysed {index} frames")
        index += 1
    cap.release()
    num_frames = index
    scores_a, disps_a = np.asarray(scores), np.asarray(disps)
    total = float(disps_a.sum())
    if min_displacement is None:
        if not target:
            raise ValueError("give target or min_displacement")
        min_displacement = total / target
    picked = select_keyframes(scores_a, disps_a, min_displacement, search_window)
    wanted = {frame_ids[i] for i in picked}
    log(f"{len(frame_ids)} frames analysed, total motion {total:.0f} px, threshold "
        f"{min_displacement:.1f} px -> {len(picked)} keyframes")

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    paths: list[Path] = []
    index = 0
    last = max(wanted)
    while index <= last:
        ok = cap.grab()
        if not ok:
            break
        if index in wanted:
            _, frame = cap.retrieve()
            if resize_width is not None and frame.shape[1] > resize_width:
                h = round(frame.shape[0] * resize_width / frame.shape[1])
                frame = cv2.resize(frame, (resize_width, h), interpolation=cv2.INTER_AREA)
            path = out_dir / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            paths.append(path)
        index += 1
    cap.release()
    stats = {
        "video": str(video_path), "fps": fps, "frames": num_frames,
        "analysed": len(frame_ids), "stride": stride,
        "analysis_width": analysis_width, "total_displacement_px": total,
        "min_displacement_px": float(min_displacement), "keyframes": [frame_ids[i] for i in picked],
        "keyframe_sharpness": [float(scores_a[i]) for i in picked],
        "median_sharpness_all": float(np.median(scores_a)),
    }
    return paths, stats
