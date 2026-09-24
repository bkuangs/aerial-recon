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
    raise NotImplementedError("M7: implement lla_to_ecef")


def ecef_to_enu(ecef: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """(N, 3) ECEF -> (N, 3) local ENU metres about the reference [lat_deg, lon_deg, alt_m]."""
    raise NotImplementedError("M7: implement ecef_to_enu")


def lla_to_enu(lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    return ecef_to_enu(lla_to_ecef(lla), ref_lla)
