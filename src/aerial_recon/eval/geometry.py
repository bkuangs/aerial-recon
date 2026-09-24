"""M9 — Geometric accuracy against a reference (LiDAR, laser scan, or synthetic GT).

The standard MVS benchmark metrics (ETH3D, Tanks and Temples, DTU):
  * precision / accuracy: how close reconstructed points are to the reference,
  * recall / completeness: how much of the reference is covered by the reconstruction,
  * F-score at threshold τ: harmonic mean of precision and recall.

Both reconstructions must be in the same metric frame first (M7 alignment). Mesh and
Gaussian outputs are compared by sampling points from their surfaces.

Reading: Knapitsch et al. "Tanks and Temples" (SIGGRAPH 2017) §4; Schöps et al. "ETH3D"
(CVPR 2017) §5; Aanæs et al. "DTU" (IJCV 2016).
"""

from __future__ import annotations

import numpy as np


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each query point (N, 3) to its nearest reference point (scipy cKDTree)."""
    raise NotImplementedError("M9: implement nearest_distances")


def geometry_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float) -> dict[str, float]:
    """Returns {"precision", "recall", "fscore", "accuracy", "completeness"}.

    precision = fraction of pred within `threshold` of gt; recall = fraction of gt within
    `threshold` of pred; fscore = 2PR / (P + R) (0 if both are 0); accuracy / completeness
    are the mean pred->gt / gt->pred distances.
    """
    raise NotImplementedError("M9: implement geometry_metrics")
