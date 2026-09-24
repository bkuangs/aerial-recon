"""GPS metadata from drone photo EXIF (DJI, Sony, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

_GPS_IFD = 0x8825


@dataclass(frozen=True)
class GpsFix:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float | None


def _dms_to_deg(dms, ref: str) -> float:
    d, m, s = (float(v) for v in dms)
    value = d + m / 60.0 + s / 3600.0
    return -value if ref in ("S", "W") else value


def read_gps(path: str | Path) -> GpsFix | None:
    """Return the GPS fix stored in an image's EXIF, or None if absent.

    Altitude is EXIF GPSAltitude (usually metres above mean sea level). DJI also writes
    *relative* (take-off) altitude and gimbal angles into XMP, which is not parsed here.
    """
    with PILImage.open(path) as img:
        gps = img.getexif().get_ifd(_GPS_IFD)
    if not gps or 2 not in gps or 4 not in gps:
        return None
    lat = _dms_to_deg(gps[2], gps.get(1, "N"))
    lon = _dms_to_deg(gps[4], gps.get(3, "E"))
    alt = None
    if 6 in gps:
        alt = float(gps[6])
        if gps.get(5, 0) in (1, b"\x01"):
            alt = -alt
    return GpsFix(lat, lon, alt)
