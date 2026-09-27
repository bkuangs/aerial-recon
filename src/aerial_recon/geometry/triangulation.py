"""M4 — Triangulation: intersecting rays that never quite meet.

Reading: HZ ch. 12 (§12.2 linear triangulation); Hartley & Sturm "Triangulation" (1997);
Lee & Civera "Triangulation: Why Optimize?" (BMVC 2019) for why the midpoint and DLT
answers differ and when it matters.
"""

from __future__ import annotations

import numpy as np


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:  # noqa: N803
    """
    Linear (DLT) triangulation of N correspondences from two 3x4 projection matrices.

    Geometrically, find the 3D point whose projections are as consistent as possible 
    with both observed pixels by casting a ray from each camera and finding where they meet.

    x1, x2 are 2D observations of the 3D world coordinates; each row is one (u, v) point.
    Row i of x1 and row i of x2 correspond to the same physical point seen from the two cameras.

    Returns 3D world coordinates.
    """
    u1, v1 = x1[:, 0:1], x1[:, 1:2]            # (N, 1) so they broadcast against (4,) rows
    u2, v2 = x2[:, 0:1], x2[:, 1:2]

    A = np.stack([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1],
    ], axis=1)                                  # (N, 4, 4)

    _, _, Vt = np.linalg.svd(A)                 # batched over the first axis
    X = Vt[:, -1]                               # (N, 4)

    return X[:, :3] / X[:, 3:]                  # homogeneous divide


def triangulate_multiview(Ps: list[np.ndarray], xs: np.ndarray) -> np.ndarray:  # noqa: N803
    """
    Linear (DLT) triangulation of a single 3D point seen in >= 2 views.

    Same idea as triangulate_dlt, but for one point tracked across many cameras: cast a ray
    from every camera through its observation and find the point closest to all of them.
    Each extra view adds two more equations, so the point gets better constrained.

    Ps is a list of V 3x4 projection matrices. xs is (V, 2): row k is the (u, v) observation
    of the point in view k, in the same coordinates Ps[k] maps into.

    Returns the 3D world coordinates, shape (3,).
    """
    A = []
    for P, (u, v) in zip(Ps, xs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])

    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]

    return X[:3] / X[3]


def triangulation_angles_deg(
    center1: np.ndarray, center2: np.ndarray, points: np.ndarray
) -> np.ndarray:
    r"""
    For each triangulated 3D point, compute the angle between the two viewing rays 
    where they meet at that point. That angle tells us how trustworthy the point's depth is.

    We define 5 to 30+° as well constrained. COLMAP's minimum is 1.5°.

   C1 •               • C2
        \           /
         \  angle  /
          \   ↓   /
           \ ⌒  /
             • X

    Small angles (< ~1-2°) mean the depth is poorly constrained: drone footage flown
    straight at a target is full of these.
    """
    r1 = center1 - points
    r2 = center2 - points

    cos_t = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))

    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))
