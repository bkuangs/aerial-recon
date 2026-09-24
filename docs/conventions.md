# Conventions

Most 3D reconstruction bugs are convention bugs. This project uses one convention
everywhere and converts at the boundaries.

## Camera frame and poses

| | This project (= OpenCV = COLMAP) | OpenGL / NeRF / Blender |
|---|---|---|
| +x | right | right |
| +y | **down** | up |
| +z | **forward** (into the scene) | backward (camera looks down −z) |
| stored pose | **world-to-camera** `x_c = R x_w + t` | usually camera-to-world `c2w` |

* The camera center is `C = −Rᵀ t` (`Pose.center`).
* `gsplat.rasterization(viewmats=...)` takes world-to-camera 4×4 matrices in the OpenCV
  convention, so `Pose.matrix()` is passed directly. Nerfstudio's `transforms.json` is
  c2w OpenGL: flip the y and z camera axes and invert.

## Pixels

* Continuous pixel coordinates follow COLMAP: **the center of the top-left pixel is
  (0.5, 0.5)**, and the image spans `[0, W] × [0, H]`.
* OpenCV (`cv2.KeyPoint.pt`, `warpPerspective`) puts pixel centers at integers. The
  scaffold converts at the boundary: `detect_sift` adds 0.5, and `mvs/warp.py`
  conjugates homographies.
* Arrays are indexed `image[row, col]`, which is `image[v, u]`. Pixel center:
  `u = col + 0.5`, `v = row + 0.5`.
* When resizing by a factor s, multiply `fx, fy, cx, cy` and all keypoints by s. With the
  0.5 convention this is exact (`Camera.scaled`).

## Rotations

* COLMAP quaternions are **(qw, qx, qy, qz)**. scipy's `Rotation.as_quat()` is **(x, y, z, w)**.
  gsplat's Gaussian quaternions are (w, x, y, z).
* Optimization uses axis-angle vectors through `so3_exp` / `so3_log`.

## World frames

* SfM output: arbitrary origin, orientation, and scale.
* Georeferenced (M7): local **ENU** (x east, y north, z up), metres, with its origin at the
  first GPS fix or scene centroid. `synthetic.py` uses this frame as well.
* GPS: WGS84 latitude/longitude in degrees. EXIF altitude is usually above mean sea level,
  not ellipsoidal. The difference, the geoid undulation, is tens of metres, but it
  doesn't matter for a local Sim3 fit.

## Depth

* "Depth" always means camera-frame **z**, not distance along the ray. Depth maps use 0 for
  invalid pixels.

## Files

* COLMAP text models (`cameras.txt`, `images.txt`, `points3D.txt`) are the interchange
  format between every tool (`io/colmap_text.py`, `modern/pose_sources.py`).
* Point clouds use PLY. Meshes use PLY/OBJ through Open3D.
* Everything generated goes under `outputs/<scene>/<method>/`, and data under `data/<scene>/`.
  Both are git-ignored.
