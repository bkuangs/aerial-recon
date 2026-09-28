"""GPS metadata from drone photo EXIF (DJI, Sony, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769

# Sensor widths (mm) for cameras whose EXIF lacks FocalPlaneXResolution. Extend as needed.
SENSOR_WIDTH_MM = {
    "FC300S": 6.17,  # DJI Phantom 3 Advanced/Pro, 1/2.3" (ODM Brighton Beach)
    "FC300X": 6.17,  # DJI Phantom 3 Professional
    "FC330": 6.17,  # DJI Phantom 4
    "FC6310": 13.2,  # DJI Phantom 4 Pro, 1"
    "DSC-WX220": 6.17,  # Sony, 1/2.3" (ODM Aukerman)
}


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


def read_focal_px(path: str | Path) -> float | None:
    """Focal length in pixels (along the image's long side) from EXIF, or None.

    Uses FocalLength with FocalPlaneXResolution when present, else the sensor-width table
    above. Deliberately does *not* use FocalLengthIn35mmFilm: vendors disagree on whether
    it refers to the diagonal or the width, and on cropped (16:9) modes.
    """
    with PILImage.open(path) as img:
        exif = img.getexif()
        sub = exif.get_ifd(_EXIF_IFD)
        width, height = img.size
    focal_mm = sub.get(0x920A)
    if not focal_mm:
        return None
    focal_mm = float(focal_mm)
    long_side = max(width, height)
    res, unit = sub.get(0xA20E), sub.get(0xA210)
    if res and unit in (2, 3):
        per_mm = float(res) / (25.4 if unit == 2 else 10.0)
        return focal_mm * per_mm
    model = str(exif.get(0x0110, "")).strip().rstrip("\x00").strip()
    sensor = SENSOR_WIDTH_MM.get(model)
    if sensor is None:
        return None
    return focal_mm / sensor * long_side
