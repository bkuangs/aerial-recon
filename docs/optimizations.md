# Optimizations in the classical pipeline (M4–M9)

Notes from building the classical pipeline and running it on Brighton Beach (18 images) and
Aukerman (77 images) on this machine (8 cores, ~7.7 GB RAM under WSL). Every number below
comes from those runs. Where an effect was not isolated, that is stated.

## Summary

| Change | Where | Measured effect |
|---|---|---|
| P3P minimal solver in PnP RANSAC | `geometry/pnp.py` | Brighton registration 2/18 → **18/18** (8-point verifier), 11/18 → 18/18 (5-point) |
| Stop trying initial pairs at 95% registered | `sfm/incremental.py` | Aukerman SfM ≈ 36 min → ≈ 12 min |
| Feature + match cache (`sfm --cache`) | `cli.py` | Reruns skip ≈ 3 min (Brighton) / ≈ 55 min (Aukerman) |
| Fusion voxel size matched to ground sampling distance | `mvs --voxel` | Aukerman fusion: OOM at 5 cm → completes at 10 cm (5.9 GB peak) |
| Bounded, shared nearest-neighbour queries + vertical pre-alignment | `eval/geometry.py`, `eval/reference.py` | Brighton evaluation: killed after > 8 min unfinished → completes |

## Accuracy and robustness

### P3P instead of DLT for PnP (the most important change)

`pnp_ransac` originally used the 12-unknown DLT (sample size 6). On Brighton Beach, the
registered 3D structure is almost planar: the smallest singular value of the centred points
is only 1–3% of the largest. Under that condition the DLT null space is not unique, so
RANSAC finds no consistent model. OpenCV's `solvePnPRansac` found 170–690 inliers on the
same correspondences, where the DLT found none.

`pnp_p3p` implements Grunert's three-point solution (a quartic, up to four poses, each
recovered with Umeyama). RANSAC samples four points, solves P3P on three, and keeps the
candidate that best reprojects the fourth. The inlier refit still uses DLT with nonlinear
refinement. In 2,000 random trials, half of them with coplanar points, the solver recovered
the pose in all but 4 near-degenerate cases.

| Brighton Beach | before (DLT) | after (P3P) |
|---|---|---|
| 8-point pair verification | 2/18 registered | 18/18, median rotation error 0.08° vs COLMAP |
| 5-point pair verification | 11/18 registered | 18/18, median rotation error 0.13° |

This changes the M6 lesson in the roadmap. The "2/18 with 8-point" gap was mostly
**planar PnP**, not a degenerate fundamental matrix.

### Other robustness choices

* **Initial pair.** Prefer pairs whose median triangulation angle is ≥ 4°, then take the one
  with the most inliers. The pair with the most matches usually has the shortest baseline.
* **Robust-scale annealing in BA.** Start the soft-L1 scale at 3× the initial RMSE, then
  tighten it to the target. A 1 px scale from a poor start barely moves (roadmap M5 pitfall).
* **PnP refinement.** Each newly registered camera is refined by nonlinear least squares
  over its inliers before triangulation.
* **Undistort once.** Every keypoint is converted to undistorted normalized coordinates at
  SfM start-up, so no estimator has to handle radial distortion.
* **Sub-plane depth refinement.** Plane sweep fits a parabola through the scores at the
  winning hypothesis and its two neighbours. Hypotheses are spaced uniformly in inverse
  depth, which gives equal pixel-disparity steps.

## Speed

### Feature and match cache

`aerial-recon sfm --cache FILE.npz` stores keypoints and every raw match. Later runs, such as
the other pose solver, the EXIF focal, or a `--sequential N` subset of the same pairs, skip
extraction and matching entirely. Without the cache, that stage took ≈ 3 min on Brighton
(153 pairs) and ≈ 55 min on Aukerman (2,926 pairs, while other jobs were running).

### Early stop on initial-pair attempts

`IncrementalSfM` tries up to three initial pairs and keeps the largest model. On Aukerman all
three reached 76/77, at ≈ 12 min each (2,142 s total). `SfMOptions.good_enough_fraction =
0.95` now stops after the first model that registers ≥ 95% of images.

### Plane sweep

* The reference image's window mean and variance are computed once and reused for every
  depth hypothesis and source view. Only the warped source needs new box filters.
* Views run in parallel in a process pool. At most 2 × workers views are in flight, to bound
  RAM.
* Throughput at 1600 px, 128 hypotheses, 4 sources, 4 workers: ≈ 13 s per view on both
  scenes (Brighton 18 views in ≈ 230 s, Aukerman 76 views in ≈ 19 min).

### Matching

`match_descriptors` uses `np.argpartition` to find the two nearest neighbours instead of a
full `argsort` per row. This change was not timed on its own. Matching is still brute force
(≈ 1 s per pair for 8k × 8k descriptors) and remains a bottleneck (see below).

### Evaluation against `model.laz`

The first `eval-geometry` run was stopped after more than 8 minutes. The two clouds were
still 10 m apart vertically, so every uncapped KD-tree query searched a long way. Fixes:

* estimate the median vertical offset on a 1 m grid first, and use it to initialise ICP;
* cap nearest-neighbour distances at 1 m (`distance_upper_bound`); the reported mean
  distances are capped accordingly;
* compute the two nearest-neighbour passes once and derive all thresholds from them;
* replace the per-point Python set lookup in the footprint crop with a boolean grid.

## Memory

### Fusion voxel size

Aukerman fusion was killed by the OOM killer twice at 5 cm voxels (peak ≈ 7.3 GB), including
after the streaming changes below. It completed at **10 cm voxels with a 5.9 GB peak**, giving
20.4M points. Pick `--voxel` near the ground sampling distance at the MVS resolution:

| Scene | Median depth | GSD at 1600 px | Voxel used |
|---|---|---|---|
| Brighton Beach | 46 m | 4.2 cm | 5 cm |
| Aukerman | 102 m | 8.6 cm | 10 cm |

Smaller voxels add memory, not detail.

### Streaming voxel accumulator

`mvs/fusion.py` keeps running per-voxel sums instead of concatenating raw points from every
view. Each view is reduced on its own first and then merged, colour sums are float32, and
grayscale images and sweep inputs are freed before fusion. These changes alone did **not**
make 5 cm Aukerman fit, and their effect at 10 cm was not measured separately.
`mvs --reuse-depth` resumes from the saved depth maps after a crash in fusion.

### Meshing

`aerial-recon mesh` works on an existing `fused.npz`, so meshing never reruns fusion.

| Scene | Input | Poisson depth | Peak RAM | Time |
|---|---|---|---|---|
| Brighton | 2.27M points | 11 | 2.5 GB | 26 s |
| Brighton | 2.27M points | 12 | 5.8 GB | 89 s |
| Aukerman | 20.4M → 4.6M points (`--voxel 0.2`) | 11 | 4.5 GB | 52 s |

## Tried and rejected

* **Capping the LSMR inner solver in BA** (`tr_options={"maxiter": 100}`). There was no
  speedup (Brighton SfM 197 s vs 194 s) at the same accuracy, so it was not kept.

## Remaining bottlenecks

Profile of Brighton SfM (8-point, from cache, 231 s total):

| Where | Time |
|---|---|
| Global bundle adjustment (9 calls) | 198 s (86%) |
| ↳ LSMR iterative linear solve inside scipy `trf` | ≈ 144 s |
| ↳ finite-difference Jacobians | ≈ 28 s |
| Pair verification (8-point RANSAC) | 27 s |

What would help, in order:

1. **A better linear solve in BA.** Solve the reduced camera system using the Schur
   complement instead of running LSMR on the full sparse Jacobian, or hand BA to Ceres
   through pycolmap. This is where most of the time goes.
2. **Local BA between global passes.** Optimise only recently registered cameras and their
   points, and run global BA less often.
3. **Analytic Jacobians.** They remove the ≈ 28 s of finite differences (roadmap M5 stretch).
4. **A faster matcher.** Use FLANN/approximate nearest neighbours or `cv2.BFMatcher`
   instead of the dense NumPy distance matrix. This matters most for exhaustive matching,
   which grows quadratically (≈ 55 min for 77 images).
5. **Vegetation in MVS.** This is quality rather than speed. Only ≈ 12% of tree-canopy depths
   survive fusion's cross-view check, against ≈ 65% for grass. Occlusion-robust aggregation
   (best-k source views) or PatchMatch with per-pixel view selection would address it.
