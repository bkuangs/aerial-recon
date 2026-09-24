"""Read/write the COLMAP text model format (cameras.txt, images.txt, points3D.txt).

This is the lingua franca of the project: COLMAP, GLOMAP, VGGT/MASt3R exporters, the
gsplat library examples, and your own SfM all meet here. Binary models can be converted
with `colmap model_converter` or `pycolmap.Reconstruction(path).write_text(out)`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aerial_recon.types import Camera, Image, Point3D, Pose, Reconstruction


def _camera_from_colmap(model: str, width: int, height: int, p: list[float]) -> Camera:
    if model == "SIMPLE_PINHOLE":
        return Camera(width, height, p[0], p[0], p[1], p[2])
    if model == "PINHOLE":
        return Camera(width, height, p[0], p[1], p[2], p[3])
    if model == "SIMPLE_RADIAL":
        return Camera(width, height, p[0], p[0], p[1], p[2], k1=p[3])
    if model == "RADIAL":
        return Camera(width, height, p[0], p[0], p[1], p[2], k1=p[3], k2=p[4])
    if model == "OPENCV":
        if abs(p[6]) > 1e-8 or abs(p[7]) > 1e-8:
            warnings.warn("OPENCV tangential distortion (p1, p2) dropped", stacklevel=2)
        return Camera(width, height, p[0], p[1], p[2], p[3], k1=p[4], k2=p[5])
    raise ValueError(f"unsupported COLMAP camera model {model}; undistort images first")


def _camera_to_colmap(cam: Camera) -> tuple[str, list[float]]:
    if not cam.has_distortion:
        return "PINHOLE", [cam.fx, cam.fy, cam.cx, cam.cy]
    if cam.fx == cam.fy:
        return "RADIAL", [cam.fx, cam.cx, cam.cy, cam.k1, cam.k2]
    return "OPENCV", [cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, 0.0, 0.0]


def rotation_to_qvec(R: np.ndarray) -> np.ndarray:  # noqa: N803
    """3x3 rotation -> COLMAP quaternion (qw, qx, qy, qz) with qw >= 0."""
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def qvec_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _data_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if not ln.startswith("#")]


def read_model(model_dir: str | Path) -> Reconstruction:
    model_dir = Path(model_dir)
    recon = Reconstruction()
    for line in _data_lines(model_dir / "cameras.txt"):
        if not line:
            continue
        tok = line.split()
        recon.cameras[int(tok[0])] = _camera_from_colmap(
            tok[1], int(tok[2]), int(tok[3]), [float(v) for v in tok[4:]])

    lines = (model_dir / "images.txt").read_text().splitlines()
    lines = [ln for ln in lines if not ln.startswith("#")]
    for header, pts in zip(lines[0::2], lines[1::2], strict=False):
        tok = header.split()
        if not tok:
            continue
        image_id = int(tok[0])
        q = np.array([float(v) for v in tok[1:5]])
        t = np.array([float(v) for v in tok[5:8]])
        p = pts.split()
        xy = np.array([float(v) for v in p], dtype=np.float64).reshape(-1, 3)
        image = Image(image_id, int(tok[8]), " ".join(tok[9:]), pose=Pose(qvec_to_rotation(q), t),
                      keypoints=xy[:, :2])
        image.point3d_ids = xy[:, 2].astype(np.int64)
        recon.images[image_id] = image

    for line in _data_lines(model_dir / "points3D.txt"):
        if not line:
            continue
        tok = line.split()
        track = np.array([int(v) for v in tok[8:]], dtype=np.int64).reshape(-1, 2)
        recon.points[int(tok[0])] = Point3D(
            xyz=np.array([float(v) for v in tok[1:4]]),
            rgb=np.array([int(v) for v in tok[4:7]], dtype=np.uint8),
            track=[(int(i), int(k)) for i, k in track],
            error=float(tok[7]),
        )
    return recon


def write_model(recon: Reconstruction, model_dir: str | Path) -> None:
    """Write only registered images (COLMAP models never contain unregistered ones)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, cam in sorted(recon.cameras.items()):
            model, params = _camera_to_colmap(cam)
            f.write(f"{cid} {model} {cam.width} {cam.height} "
                    + " ".join(f"{v:.12g}" for v in params) + "\n")

    registered = set(recon.registered_image_ids)
    with open(model_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for iid in sorted(registered):
            im = recon.images[iid]
            q = rotation_to_qvec(im.pose.R)
            f.write(f"{iid} " + " ".join(f"{v:.12g}" for v in (*q, *im.pose.t))
                    + f" {im.camera_id} {im.name}\n")
            f.write(" ".join(f"{x:.6f} {y:.6f} {int(p)}"
                             for (x, y), p in zip(im.keypoints, im.point3d_ids, strict=True))
                    + "\n")

    with open(model_dir / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, pt in sorted(recon.points.items()):
            track = [(i, k) for i, k in pt.track if i in registered]
            if not track:
                continue
            f.write(f"{pid} " + " ".join(f"{v:.12g}" for v in pt.xyz) + " "
                    + " ".join(str(int(v)) for v in pt.rgb) + f" {pt.error:.6g} "
                    + " ".join(f"{i} {k}" for i, k in track) + "\n")
