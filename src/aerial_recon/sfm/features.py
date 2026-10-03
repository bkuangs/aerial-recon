"""M3 — Local features and descriptor matching.

Detection/description is scaffolded with OpenCV SIFT (implementing SIFT from scratch is a
worthwhile side quest but not the point here). Matching is yours: it decides what
geometry downstream has to survive.

Keypoints are returned in COLMAP pixel convention (top-left pixel center = (0.5, 0.5));
OpenCV uses (0, 0), hence the +0.5 below. Getting this wrong is a classic half-pixel bug.

Reading: Lowe "Distinctive Image Features from Scale-Invariant Keypoints" (IJCV 2004) §6–7;
Arandjelović & Zisserman "Three things everyone should know..." (RootSIFT, CVPR 2012);
Szeliski 2e §7.1.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_sift(gray: np.ndarray, max_features: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """SIFT keypoints (N, 2) in COLMAP convention and float32 descriptors (N, 128).

    Like COLMAP, detect everything at a low contrast threshold and keep the `max_features`
    *largest-scale* keypoints. OpenCV's own `nfeatures` keeps the strongest responses
    instead, which on aerial imagery means fine sand/grass/wave texture that doesn't
    repeat across views: on Brighton Beach that choice cut matches per pair by ~3x.
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    sift = cv2.SIFT_create(contrastThreshold=0.02)
    kps = sift.detect(gray, None)
    kps = sorted(kps, key=lambda kp: -kp.size)[:max_features]
    kps, desc = sift.compute(gray, kps)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2)), np.zeros((0, 128), dtype=np.float32)
    xy = np.array([kp.pt for kp in kps], dtype=np.float64) + 0.5
    return xy, desc.astype(np.float32)


def root_sift(descriptors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RootSIFT: L1-normalize each descriptor, then take the element-wise square root.

    Euclidean distance between RootSIFT vectors equals the Hellinger kernel on the
    originals, which matches noticeably better. Output rows have unit L2 norm.
    """
    l1 = np.sum(np.abs(descriptors), axis=1, keepdims=True)
    return np.sqrt(descriptors / (l1 + eps))


def match_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio: float = 0.8,
    mutual: bool = True,
) -> np.ndarray:
    """
    Nearest-neighbour distance ratio test with an optional mutual check.

    * Ratio test: keep i -> j only if d(i, nn1) < ratio * d(i, nn2) (Euclidean distances).
    * Mutual check: additionally require that i is also b[j]'s nearest neighbour in a.

    Returns (M, 2) int64 index pairs (i into desc_a, j into desc_b).
    """
    if len(desc_a) == 0 or desc_b.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.int64)

    # Euclidean distances
    aa = np.sum(desc_a**2, axis=1)[:, None]    # (Na, 1)
    bb = np.sum(desc_b**2, axis=1)[None, :]    # (1, Nb)
    d2 = aa + bb - 2 * desc_a @ desc_b.T       # (Na, Nb)
    d = np.sqrt(np.maximum(d2, 0))             # clamp tiny negatives from rounding

    # Two nearest neighbours in O(Nb) per row (a full argsort is O(Nb log Nb)).
    nn = np.argpartition(d, 1, axis=1)[:, :2]
    rows = np.arange(len(desc_a))
    swap = d[rows, nn[:, 0]] > d[rows, nn[:, 1]]
    nn[swap] = nn[swap][:, ::-1]
    d1, d2nd = d[rows, nn[:, 0]], d[rows, nn[:, 1]]
    keep = d1 < ratio * d2nd
    j = nn[:, 0]

    # Cross-check: A match is kept only if the two descriptors pick each other both ways
    if mutual:
        best_a_for_b = np.argmin(d, axis=0)
        keep &= best_a_for_b[j] == rows

    return np.column_stack([rows[keep], j[keep]]).astype(np.int64)
