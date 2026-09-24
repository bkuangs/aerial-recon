"""M4 — Triangulation: intersecting rays that never quite meet.

Reading: HZ ch. 12 (§12.2 linear triangulation); Hartley & Sturm "Triangulation" (1997);
Lee & Civera "Triangulation: Why Optimize?" (BMVC 2019) for why the midpoint and DLT
answers differ and when it matters.
"""

from __future__ import annotations

import numpy as np


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:  # noqa: N803
    """Linear (DLT) triangulation of N correspondences from two 3x4 projection matrices.

    x1, x2 are (N, 2) in the same coordinates the P's map into (pixels if P = K[R|t],
    normalized if P = [R|t]). Returns (N, 3). Vectorize over N (np.linalg.svd batches).
    """
    raise NotImplementedError("M4: implement triangulate_dlt")


def triangulate_multiview(Ps: list[np.ndarray], xs: np.ndarray) -> np.ndarray:  # noqa: N803
    """DLT triangulation of a single point seen in V >= 2 views. xs is (V, 2); returns (3,)."""
    raise NotImplementedError("M4: implement triangulate_multiview")


def triangulation_angles_deg(
    center1: np.ndarray, center2: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Angle at each 3D point (N, 3) between the rays to the two camera centers, degrees.

    Small angles (< ~1-2°) mean the depth is poorly constrained: drone footage flown
    straight at a target is full of these.
    """
    raise NotImplementedError("M4: implement triangulation_angles_deg")
