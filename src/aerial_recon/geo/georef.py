"""M7 gate: georeference a reconstruction with EXIF GPS (camera centers -> local ENU).

    EXIF GPS (WGS84) --lla_to_enu--> ENU centers --umeyama--> Sim3 --> transform model

The reference point of the ENU frame is the mean GPS fix, so every later stage (MVS,
evaluation against ODM's UTM point cloud) can reproduce the frame from `georef.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerial_recon.geo.geodesy import lla_to_enu
from aerial_recon.geometry.alignment import Sim3, transform_reconstruction, umeyama
from aerial_recon.io.exif import read_gps
from aerial_recon.types import Reconstruction


@dataclass
class GeorefResult:
    sim3: Sim3
    ref_lla: np.ndarray  # (3,) lat, lon, alt of the ENU origin
    names: list[str]
    residuals: np.ndarray  # (N, 3) ENU residuals (aligned - GPS), metres
    gps_enu: np.ndarray  # (N, 3) GPS positions in ENU, metres

    def summary(self) -> dict:
        r = self.residuals
        horiz = np.linalg.norm(r[:, :2], axis=1)
        return {
            "num_images": len(self.names),
            "scale": float(self.sim3.s),
            "rmse_3d_m": float(np.sqrt(np.mean(np.sum(r**2, axis=1)))),
            "rmse_horizontal_m": float(np.sqrt(np.mean(horiz**2))),
            "rmse_vertical_m": float(np.sqrt(np.mean(r[:, 2] ** 2))),
            "max_3d_m": float(np.linalg.norm(r, axis=1).max()),
            **trend_diagnostics(self),
        }


def trend_diagnostics(result: GeorefResult) -> dict:
    """Systematic-error checks on GPS residuals.

    * vertical-vs-radius slope: doming shows up as vertical residual growing (or
      shrinking) with horizontal distance from the block center.
    * R² of a linear fit of residuals on (E, N): a large value means a tilt/scale error
      rather than GPS noise.
    """
    gps = result.gps_enu
    r = result.residuals
    xy = gps[:, :2] - gps[:, :2].mean(axis=0)
    radius = np.linalg.norm(xy, axis=1)
    out = {}
    if np.ptp(radius) > 1e-9:
        slope = np.polyfit(radius, r[:, 2], 1)[0]
        out["vertical_residual_vs_radius_slope"] = float(slope)
    A = np.column_stack([xy, np.ones(len(xy))])  # noqa: N806
    r2 = []
    for k in range(3):
        coef, *_ = np.linalg.lstsq(A, r[:, k], rcond=None)
        ss_res = np.sum((r[:, k] - A @ coef) ** 2)
        ss_tot = np.sum((r[:, k] - r[:, k].mean()) ** 2)
        r2.append(float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0)
    out["residual_planar_trend_r2_enu"] = r2
    return out


def gps_enu_for_images(
    recon: Reconstruction, image_dir: str | Path, ref_lla: np.ndarray | None = None
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(names, ENU (N, 3), ref_lla) for registered images that carry an EXIF GPS fix."""
    image_dir = Path(image_dir)
    names, lla = [], []
    for iid in recon.registered_image_ids:
        name = recon.images[iid].name
        fix = read_gps(image_dir / name)
        if fix is None or fix.altitude_m is None:
            continue
        names.append(name)
        lla.append([fix.latitude_deg, fix.longitude_deg, fix.altitude_m])
    lla = np.asarray(lla, dtype=np.float64)
    if len(lla) < 3:
        raise ValueError(f"need >= 3 registered images with GPS, found {len(lla)}")
    if ref_lla is None:
        ref_lla = lla.mean(axis=0)
    return names, lla_to_enu(lla, ref_lla), np.asarray(ref_lla, dtype=np.float64)


def georeference(recon: Reconstruction, image_dir: str | Path) -> GeorefResult:
    """Sim3-align camera centers to GPS (ENU) and transform `recon` **in place**."""
    names, enu, ref_lla = gps_enu_for_images(recon, image_dir)
    by_name = {im.name: im for im in recon.images.values() if im.is_registered}
    centers = np.stack([by_name[n].pose.center for n in names])
    sim3 = umeyama(centers, enu, with_scale=True)
    transform_reconstruction(recon, sim3)
    aligned = np.stack([by_name[n].pose.center for n in names])
    return GeorefResult(sim3, ref_lla, names, aligned - enu, enu)


def write_georef(result: GeorefResult, path: str | Path) -> dict:
    summary = result.summary()
    payload = {
        "ref_lla": result.ref_lla.tolist(),
        "frame": "local ENU (x east, y north, z up) in metres about ref_lla; "
                 "altitude datum = EXIF GPSAltitude (usually MSL)",
        "sim3": {"s": result.sim3.s, "R": result.sim3.R.tolist(), "t": result.sim3.t.tolist()},
        "summary": summary,
        "residuals_enu_m": {n: r.tolist() for n, r in zip(result.names, result.residuals,
                                                          strict=True)},
    }
    Path(path).write_text(json.dumps(payload, indent=2))
    return summary


def read_ref_lla(path: str | Path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text())["ref_lla"], dtype=np.float64)
