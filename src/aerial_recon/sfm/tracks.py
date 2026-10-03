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
        self.parent: dict = {}
        self.rank: dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


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
    uf = UnionFind()
    for (a, b), m in matches.items():
        for i, j in np.asarray(m, dtype=np.int64).reshape(-1, 2):
            uf.union((int(a), int(i)), (int(b), int(j)))

    components: dict = {}
    for node in list(uf.parent):
        components.setdefault(uf.find(node), []).append(node)

    tracks = []
    for nodes in components.values():
        images = [image_id for image_id, _ in nodes]
        if len(images) != len(set(images)):  # conflicting track: drop it
            continue
        if len(nodes) >= min_length:
            tracks.append(sorted(nodes))
    tracks.sort(key=lambda t: t[0])
    return tracks
