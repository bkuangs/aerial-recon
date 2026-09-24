"""M6 — Feature tracks: stitching pairwise matches into multi-view observations.

If keypoint a3 in image A matches b7 in B, and b7 matches c1 in C, they are (probably) the
same 3D point. Connected components of the match graph are tracks. A component that
contains two different keypoints from the same image is inconsistent (a bad match
somewhere) and is discarded.

Reading: Moulon & Monasse "Unordered feature tracking made fast and easy" (CVMP 2012);
Szeliski 2e §11.4.
"""

from __future__ import annotations

import numpy as np


class UnionFind:
    """Disjoint-set forest with path compression and union by rank over hashable nodes."""

    def __init__(self) -> None:
        raise NotImplementedError("M6: implement UnionFind")

    def find(self, x):
        raise NotImplementedError

    def union(self, a, b) -> None:
        raise NotImplementedError


def build_tracks(
    matches: dict[tuple[int, int], np.ndarray], min_length: int = 2
) -> list[list[tuple[int, int]]]:
    """Connected components of {(image_id, keypoint_idx)} nodes linked by matches.

    Args:
        matches: {(image_a, image_b): (M, 2) keypoint index pairs}.
    Returns:
        Tracks as lists of (image_id, keypoint_idx), each sorted, containing at most one
        keypoint per image, with length >= min_length. Tracks are sorted by their first
        element so the output is deterministic.
    """
    raise NotImplementedError("M6: implement build_tracks")
