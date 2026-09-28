"""M9 real-data evaluation: compare a fused cloud against a reference cloud (LiDAR / ODM).

Both clouds must share a metric frame. The prediction is in the local ENU frame written by
`aerial-recon georef` (GPS-aligned, so only good to a few metres). ODM's `model.laz` is in
UTM. Evaluation therefore:

  1. converts the reference to the same ENU frame (pyproj: UTM -> WGS84 -> ENU),
  2. crops the prediction to the reference's footprint (a 2D occupancy grid), because
     precision should not be charged for areas the reference simply doesn't cover,
  3. optionally refines the GPS alignment with trimmed ICP (as Tanks and Temples does),
  4. reports precision / recall / F-score at each threshold, before and after ICP.

ODM's cloud is itself photogrammetric, so report agreement, not accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from aerial_recon.eval.geometry import geometry_metrics
from aerial_recon.geo.geodesy import lla_to_enu
from aerial_recon.geometry.alignment import Sim3, umeyama


def load_laz_enu(path: str | Path, ref_lla: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """LAS/LAZ (projected CRS from its header) -> ENU points about `ref_lla`, and RGB."""
    import laspy
    from pyproj import Transformer

    las = laspy.read(str(path))
    crs = las.header.parse_crs()
    if crs is None:
        raise ValueError(f"{path} has no CRS in its header")
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(xyz[:, 0], xyz[:, 1])
    enu = lla_to_enu(np.column_stack([lat, lon, xyz[:, 2]]), ref_lla)
    rgb = None
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        rgb = (rgb / (257.0 if rgb.max() > 255 else 1.0)).astype(np.uint8)
    return enu, rgb


def footprint_mask(points: np.ndarray, reference: np.ndarray, cell: float = 1.0) -> np.ndarray:
    """Points whose (x, y) cell is occupied by the reference."""
    lo = reference[:, :2].min(axis=0)
    ref_cells = np.floor((reference[:, :2] - lo) / cell).astype(np.int64)
    shape = ref_cells.max(axis=0) + 1
    grid = np.zeros(shape, dtype=bool)
    grid[ref_cells[:, 0], ref_cells[:, 1]] = True
    cells = np.floor((points[:, :2] - lo) / cell).astype(np.int64)
    inside = (cells >= 0).all(axis=1) & (cells < shape).all(axis=1)
    mask = np.zeros(len(points), dtype=bool)
    mask[inside] = grid[cells[inside, 0], cells[inside, 1]]
    return mask


def icp(source: np.ndarray, target: np.ndarray, max_distance: float = 2.0,
        min_distance: float = 0.1, iterations: int = 40, with_scale: bool = False,
        sample: int = 200_000, seed: int = 0) -> Sim3:
    """Trimmed point-to-point ICP (Umeyama per iteration) that anneals the inlier radius
    from `max_distance` down to `min_distance`. Returns source -> target."""
    rng = np.random.default_rng(seed)
    src = source[rng.choice(len(source), min(sample, len(source)), replace=False)]
    tree = cKDTree(target)
    total = Sim3(1.0, np.eye(3), np.zeros(3))
    radii = np.geomspace(max_distance, min_distance, iterations)
    for radius in radii:
        moved = total.apply(src)
        d, idx = tree.query(moved, k=1, distance_upper_bound=radius, workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 100:
            break
        step = umeyama(moved[ok], target[idx[ok]], with_scale=with_scale)
        total = Sim3(step.s * total.s, step.R @ total.R, step.s * step.R @ total.t + step.t)
    return total


def evaluate_against_reference(pred: np.ndarray, reference: np.ndarray,
                               thresholds: list[float], use_icp: bool = True,
                               icp_scale: bool = False) -> dict:
    inside = footprint_mask(pred, reference)
    cropped = pred[inside]
    out: dict = {"num_pred": int(len(pred)), "num_pred_in_footprint": int(inside.sum()),
                 "num_reference": int(len(reference))}
    out["gps_aligned"] = {f"{t:g}": geometry_metrics(cropped, reference, t) for t in thresholds}
    if use_icp:
        T = icp(cropped, reference, with_scale=icp_scale)  # noqa: N806
        aligned = T.apply(cropped)
        angle = float(np.degrees(np.arccos(np.clip((np.trace(T.R) - 1) / 2, -1, 1))))
        out["icp"] = {"translation_m": T.t.tolist(), "rotation_deg": angle, "scale": T.s}
        out["icp_aligned"] = {f"{t:g}": geometry_metrics(aligned, reference, t)
                              for t in thresholds}
    return out
