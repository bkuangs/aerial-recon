# Optimizations in the classical pipeline (M4–M9)

Notes from building the classical pipeline and running it on Brighton Beach (18 images) and
Aukerman (77 images) on this machine (8 cores, ~7.7 GB RAM under WSL, CPU only), and later
on two drone-orbit videos ([last section](#video-reconstruction-drone-orbits)). Every
number below comes from those runs. Where an effect was not isolated, that is stated.

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

## Video reconstruction (drone orbits)

Two 2560×1440, 60 fps orbits: `drone_roof_orbit` (8,673 frames) and
`warehouse_drone_orbit` (8,234 frames). Pipeline: `keyframes` → COLMAP → `level` → `mvs`.
The models have no GPS, so they are in arbitrary units.

| Change | Where | Measured effect |
|---|---|---|
| Keyframe selection before SfM | `video/frames.py`, `keyframes` | 8,673 → 165 and 8,234 → 156 frames; exhaustive COLMAP in 13 and 8 min |
| Two-pass decode, analysis at 640 px | `extract_keyframes` | ≈ 112 s per video (both at once), 0.3 GB RSS |
| Fusion voxel = 4 × GSD, not 1 × | `mvs --voxel 0`, `--auto-voxel-gsd` | Roof fusion: OOM at 1 GSD → completes at 4 GSD, 4.1 GB peak, 3.9M points |
| Buffered, sorted voxel store | `mvs/fusion.py` | Effect not isolated from the voxel change (see below) |
| `mvs --reuse-depth` after a fusion crash | `mvs/pipeline.py` | Each retry skipped ≈ 28 min of plane sweep (twice on the roof) |

### Keyframes first, then exhaustive matching

Feeding every frame to SfM is out of reach on a CPU. Exhaustive matching of 8,673 frames
would mean about 37.6M pairs. `keyframes` keeps about one frame per 0.86 s. It scores
sharpness (Laplacian variance) and motion (median LK flow of 400 corners) on 640 px
grayscale frames. The motion threshold is total flow / `--target`. With about 160
keyframes, **exhaustive** matching is affordable: 13,530 pairs for the roof, with COLMAP
taking 13 min in total and peaking at 5.0 GB. Exhaustive matching was chosen over sequential
matching because an orbit returns to where it started. Sequential matching only closes
that loop with vocabulary-tree loop detection, and no vocabulary tree was set up here.
Every keyframe registered (165/165 and 156/156).

The decoder runs twice. Pass 1 decodes and scores every frame at 640 px. Pass 2 calls
`cap.grab()` for every frame and `retrieve()`s only the selected ones. Frames are written
at 1920 px (COLMAP's `--max-size`), not 2560. The saving from the smaller output was not
measured.

### Voxel size in a scale-free model

`georef` cannot run without GPS, so there is no metric voxel size. `mvs --voxel 0` uses
`auto_voxel_gsd` × the median ground sampling distance, measured from the sparse points.
One GSD was the Brighton/Aukerman rule, and it fails for an orbit:

| Roof fusion attempt | Result |
|---|---|
| 1 GSD (0.004), original accumulator | OOM-killed after ≈ 14 min of fusion, 6.7 GB peak |
| 1 GSD, buffered accumulator (below) | 5.2 GB RSS after 11 min and still rising; stopped |
| **4 GSD (0.016)**, buffered accumulator | **completes in 16 min, 4.1 GB peak, 3.9M points** |

The reason is redundancy. All 165 views see the same buildings, so each surface point is
back-projected many times. Depth noise then spreads it through a shell many GSDs thick,
because at these short baselines both the plane-sweep depth steps (before sub-plane
refinement) and the 1% relative-depth tolerance of the consistency check are many GSDs
deep. At a depth of 5 units, 1% is 0.05, or 12 GSD. At 1 GSD almost every raw point gets
its own voxel, so voxelisation stops reducing anything. The warehouse ran at 4 GSD
straight away: 10 min fusion, 3.6 GB peak, 2.9M points.

### Buffered voxel accumulator

The original accumulator ran `np.unique(..., return_inverse=True)` over the *whole* store
after every view. That re-sorted tens of millions of keys 165 times and briefly held
several full-size temporaries. The rewrite does three things:

* Each view is reduced on its own (sort plus `np.add.reduceat`) and buffered.
* Buffers are merged into a **sorted** store only once they reach max(4M, store / 4)
  entries. Existing voxels are found with `np.searchsorted` and added to in place. New ones
  are placed with `np.insert`.
* Sums are float32 offsets from the voxel corner (sub-voxel values, so float32 is ample),
  plus int64 keys and int32 counts: 32 B per voxel, 44 B with colour.

On 2.4M random points it matched a one-shot reduction to 5e-9 in position.
`test_voxel_accumulator_buffered_merge_matches_one_shot` checks the same property. Its own
effect was not isolated. At 1 GSD it did not fit either, so the voxel size was the
change that decided it.

### Remaining bottlenecks (video)

| Stage (roof, 165 views at 1600×900) | Time |
|---|---|
| Plane sweep, 4 workers | 28 min (≈ 10 s per view) |
| Fusion consistency check + voxel merge | 16 min (≈ 6 s per view) |
| COLMAP features + exhaustive matching + mapping | 13 min |

Fusion now costs more than half as much as depth estimation. Each view reprojects every
valid pixel into 8 neighbours in float64 NumPy, in the main process. Running it in float32
or across the process pool are the obvious next steps (not tried). The CUDA
paths (COLMAP matching, PatchMatch stereo) were unavailable: WSL reports
`CUDA_ERROR_OPERATING_SYSTEM`, and the pip `pycolmap` build is CPU-only.
