# Datasets

Tiered from "always available, exact answers" to "large, real, with ground truth". Start
at the top, and only move down once the tier above works.

| Tier | Dataset | What | Size | Ground truth | Use in |
|------|---------|------|------|--------------|--------|
| 0 | `synthetic.py` (built in) | orbits, survey grids, terrain, rooftop slabs | — | exact poses, points, depth | every student test |
| 1 | **ODM Brighton Beach** | 18 DJI images, oblique + nadir, BSD-2 | ~80 MB | ODM DSM + point cloud (pseudo-reference) | inner dev loop, M6–M11 |
| 1 | **ODM Aukerman** | 77 nadir survey images over a park with buildings, CC0 | ~520 MB | none (COLMAP as reference) | M6 at scale, M9, study |
| 1 | ODM Toledo | 87 DJI survey images | ~430 MB | none | study, second scene |
| 1 | ODM Pacifica | 12 images | ~90 MB | none | failure-mode experiments |
| 2 | **ETH3D high-res multi-view** | DSLR, indoor + outdoor, eth3d.net | 1–10 GB | laser scans | M8–M9 accuracy (not aerial) |
| 2 | DTU (already at `~/gsplat/data/dtu`) | object-scale, structured light | — | GT points | MVS sanity checks |
| 3 | **UrbanScene3D** (vcc.tech/UrbanScene3D) | drone imagery of urban scenes | large | LiDAR for some scenes | study ground truth |
| 3 | **Mill-19** (Mega-NeRF: Building, Rubble) | ~2k high-res drone images per scene | large | none | large-scale 3DGS stretch |
| 3 | **MatrixCity** (city-super.github.io/matrixcity) | synthetic UE city, aerial + street | very large | exact poses and depth | pose/depth evaluation (download one block) |

`aerial-recon download list` shows the Tier 1 sets, which download automatically. Tiers 2–3
need manual download or registration. Check each license before publishing renders.

## Why these choices

* **Brighton Beach** is small enough for a full SfM → MVS → 3DGS round trip in minutes.
  It's also flat and textureless, so it stresses the planar degeneracy (M4/M6) and weak
  features (M3). The ODM `model.laz` and `dsm.tif` give a geometric reference, but it's
  photogrammetric, not LiDAR, so report it as agreement, not accuracy.
* **Aukerman** is the classic nadir survey: many images, strong overlap, buildings, and
  trees. It's where doming, sequential vs exhaustive matching, and memory limits show up.
* **ETH3D / DTU** give the MVS track true ground truth before you trust any aerial F-score.
* **UrbanScene3D / MatrixCity** are where the study's geometry claims become defensible.

## Video

You don't have a drone yet, so for M2 and the video → reconstruction story:

1. **Wikimedia Commons, Category:Aerial videos.** Many CC-BY/CC-BY-SA drone clips. Pick
   a slow orbit around a building. Record the file's license in `data/<name>/LICENSE.txt`.
2. **Synthetic video.** Render a dense orbit with `synthetic.render_slab_scene` (exact
   poses) to test keyframe selection against known baselines.
3. **Your own footage later.** DJI writes per-frame telemetry to `.SRT` sidecar files
   (GPS, altitude, gimbal). A parser for it is a natural M7 stretch.

Suggested flight for your first capture: a 360° orbit at a constant 30–50 m altitude with
the gimbal at −30° to −45° and 60–80% overlap between keyframes, plus one nadir
lawnmower pass. That gives both viewpoint types the study compares.
