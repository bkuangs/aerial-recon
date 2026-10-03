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
from scipy.spatial import cKDTree


def nearest_distances(query: np.ndarray, reference: np.ndarray,
                      max_distance: float = np.inf) -> np.ndarray:
    """Distance from each query point (N, 3) to its nearest reference point (scipy cKDTree).

    Distances beyond `max_distance` are returned as `max_distance`. Bounding the search is
    much faster when the clouds are far apart (e.g. before fine alignment).
    """
    d, _ = cKDTree(np.asarray(reference, dtype=np.float64)).query(
        np.asarray(query, dtype=np.float64), k=1, workers=-1, distance_upper_bound=max_distance)
    return np.minimum(d, max_distance)


def geometry_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float,
                     max_distance: float = np.inf) -> dict[str, float]:
    """Returns {"precision", "recall", "fscore", "accuracy", "completeness"}.

    precision = fraction of pred within `threshold` of gt; recall = fraction of gt within
    `threshold` of pred; fscore = 2PR / (P + R) (0 if both are 0); accuracy / completeness
    are the mean pred->gt / gt->pred distances (capped at `max_distance` if given).
    """
    return geometry_metrics_multi(pred, gt, [threshold], max_distance)[threshold]


def geometry_metrics_multi(pred: np.ndarray, gt: np.ndarray, thresholds: list[float],
                           max_distance: float = np.inf) -> dict[float, dict[str, float]]:
    """geometry_metrics for several thresholds, sharing the two nearest-neighbour passes."""
    d_pred = nearest_distances(pred, gt, max_distance)
    d_gt = nearest_distances(gt, pred, max_distance)
    return {t: _metrics(d_pred, d_gt, t) for t in thresholds}


def _metrics(d_pred: np.ndarray, d_gt: np.ndarray, threshold: float) -> dict[str, float]:
    precision = float(np.mean(d_pred < threshold))
    recall = float(np.mean(d_gt < threshold))
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "fscore": float(fscore),
            "accuracy": float(np.mean(d_pred)), "completeness": float(np.mean(d_gt))}
