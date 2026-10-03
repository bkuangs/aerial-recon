"""M7 — Geodesy: GPS latitude/longitude/altitude to a local metric frame.

GPS gives WGS84 geodetic coordinates. Reconstruction wants a local Cartesian frame in
metres. The standard route: geodetic -> ECEF (Earth-centred, Earth-fixed) -> ENU (East,
North, Up) tangent plane at a reference point near the scene.

WGS84: a = 6378137.0 m, f = 1 / 298.257223563, e² = f (2 - f).

Reading: "Geographic coordinate conversion" (Wikipedia, the ECEF and ENU sections are
accurate); Hofmann-Wellenhof et al. "GNSS" ch. 8 if you want the real thing.
Drone GPS is typically 1-5 m accurate horizontally and worse vertically; barometric
altitude drifts. Treat it as a weak prior, not ground truth.
"""

from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla_to_ecef(lla: np.ndarray) -> np.ndarray:
    """(N, 3) [lat_deg, lon_deg, alt_m (ellipsoidal)] -> (N, 3) ECEF metres."""
    lla = np.asarray(lla, dtype=np.float64).reshape(-1, 3)
    lat, lon, h = np.deg2rad(lla[:, 0]), np.deg2rad(lla[:, 1]), lla[:, 2]
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)  # prime vertical radius
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * np.sin(lat)
    return np.column_stack([x, y, z])


def enu_rotation(ref_lla: np.ndarray) -> np.ndarray:
    """Rows are the east, north, up unit vectors (in ECEF) at the reference point."""
    lat, lon = np.deg2rad(ref_lla[0]), np.deg2rad(ref_lla[1])
    return np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])


def ecef_to_enu(ecef: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """(N, 3) ECEF -> (N, 3) local ENU metres about the reference [lat_deg, lon_deg, alt_m]."""
    ref_lla = np.asarray(ref_lla, dtype=np.float64).reshape(3)
    origin = lla_to_ecef(ref_lla[None])[0]
    return (np.asarray(ecef, dtype=np.float64).reshape(-1, 3) - origin) @ enu_rotation(ref_lla).T


def lla_to_enu(lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    return ecef_to_enu(lla_to_ecef(lla), ref_lla)
