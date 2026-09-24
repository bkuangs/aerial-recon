"""Minimal ASCII PLY writer/reader for point clouds (open in MeshLab / CloudCompare)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format ascii 1.0", f"element vertex {len(points)}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = np.clip(np.round(colors * 255 if colors.max() <= 1.0 else colors), 0, 255)
        colors = colors.astype(np.uint8).reshape(-1, 3)
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i, p in enumerate(points):
            row = f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
            if colors is not None:
                row += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            f.write(row + "\n")


def read_ply_points(path: str | Path) -> np.ndarray:
    """Read xyz from an ASCII PLY written by `write_ply` (use open3d for anything else)."""
    lines = Path(path).read_text().splitlines()
    end = lines.index("end_header")
    n = next(int(ln.split()[-1]) for ln in lines[:end] if ln.startswith("element vertex"))
    data = np.array([[float(v) for v in ln.split()[:3]] for ln in lines[end + 1:end + 1 + n]])
    return data.reshape(-1, 3)
