# Roadmap

Five phases, fourteen milestones. Each milestone has a **learning goal**, a **build**
list, a **gate** (an objective definition of done), and **pitfalls** found while building
the reference solutions. Rough effort assumes hobby pace, in focused sessions of a few
hours each.

Scope discipline, borrowed from `~/gsplat`:

* **Must finish:** M0–M7 (a working SfM you understand) and M10–M11 (splats on drone data).
* **Target:** M8–M9 (your own MVS) and M13 (a comparison with a clear answer).
* **Stretch:** PatchMatch, M12, large-scale tiling, and your own drone footage. Don't
  commit to these in advance.

---

## Phase A: Foundations

### M0: Bringup and "see the answer first" (1 session)

**Goal:** Know what the finished classical output looks like before you build it.

* `uv sync --extra colmap --extra dev` then `uv run pytest tests/scaffold`.
* `aerial-recon download brighton_beach` then `aerial-recon colmap data/brighton_beach/images --out outputs/brighton_beach/colmap`.
* Open `outputs/brighton_beach/colmap/sparse/0` in the COLMAP GUI (Windows build), or
  load `points3D` into MeshLab/CloudCompare. Also try `--mapper global` (GLOMAP) and
  compare runtimes.
* Read [conventions.md](conventions.md) end to end. Every later bug is a convention bug
  until proven otherwise.

**Gate:** Screenshot of the sparse model with camera frusta. You should be able to explain
what each file in `sparse_txt/` contains.

### M1: Camera geometry and SO(3) (2 sessions)

**Goal:** Projection, distortion, and rotation parameterization that you can derive on paper.
**Build:** `geometry/rotations.py` (hat, exp, log, projection to SO(3), angle) and
`geometry/camera.py` (distort/undistort, project/unproject, reprojection errors).
**Gate:** `pytest tests/student -k m01`.
**Pitfalls:** `so3_log` near π: the tests go to within 1e-7 rad of π. Undistortion has no
closed form, so iterate.

### M2: Video → keyframes (1–2 sessions)

**Goal:** Turn a 30 fps video into a sharp, well-spaced image set.
**Build:** `video/frames.py`: sharpness (variance of Laplacian), LK-flow displacement,
and deterministic keyframe selection.
**Gate:** `-k m02`, then on a real aerial clip (see [datasets.md](datasets.md#video)):
extract every 2nd frame, score, select, and run the COLMAP reference with
`--matcher sequential`. Compare registration against naive every-Nth-frame sampling.
**Stretch:** Reject frames by rolling-shutter-prone angular velocity, or by matching
overlap instead of flow.

---

## Phase B: Classical structure-from-motion

### M3: Features, matching, RANSAC, homographies (2 sessions)

**Build:** `root_sift` and `match_descriptors` (ratio test and mutual check),
`geometry/ransac.py` (adaptive iterations and refit), and `geometry/homography.py`
(normalized DLT).
**Gate:** `-k m03`. This includes a real SIFT → match → RANSAC run on rendered nadir views.
**Pitfalls:**
* `required_iterations`: `log(1 - p)` rounds to 0 when p ≈ 1e-18. Use `log1p`.
* Degenerate minimal samples produce singular homographies. Return `None`, don't crash.
* OpenCV SIFT's `nfeatures` keeps the *strongest* keypoints. On aerial scenes that means
  sand, grass, and wave texture. The scaffold keeps the *largest-scale* keypoints like
  COLMAP does, which roughly tripled matches on Brighton Beach.

### M4: Two-view geometry and triangulation (2–3 sessions)

**Build:** Normalized 8-point, Sampson distance, E from F, the four-fold decomposition,
cheirality, robust relative pose, DLT and multi-view triangulation, and triangulation angles.
**Gate:** `-k m04`.
**Exercise:** Take two nadir images over flat ground and estimate both F and H. Why is F
unreliable? This comes back hard in M6.

### M5: Bundle adjustment (2–3 sessions)

**Build:** `sfm/bundle_adjustment.py`: axis-angle pose parameterization, the sparsity
pattern, a robust loss, motion-only mode, and fixed cameras.
**Gate:** `-k m05`.
**Pitfalls (both found while building the reference):**
* A robust loss with a 1 px scale stalls from a distant start. Anneal the scale down.
* scipy's `huber` converged pathologically slowly on these problems, while `soft_l1`
  took about 20 evaluations. Default is `soft_l1`, and your notes should explain why.
* Finite-difference Jacobians work but dominate runtime on real scenes. Analytic Jacobians
  are the first optimization to make.

### M6: Incremental SfM (3–5 sessions, the big one)

**Build:** `geometry/pnp.py` (DLT and RANSAC), `sfm/tracks.py` (union-find, conflict
rejection), and `sfm/incremental.py` (verify pairs, choose the initial pair, register,
triangulate, filter, periodic BA).
**Gate (synthetic):** `-k m06`: a 16-view orbit plus a 3×6 oblique survey grid with 30%
outlier matches.
**Gate (real):**
```bash
aerial-recon sfm data/brighton_beach/images --intrinsics outputs/brighton_beach/colmap/sparse_txt \
    --out outputs/brighton_beach/mine [--five-point]
aerial-recon compare-poses outputs/brighton_beach/mine outputs/brighton_beach/colmap/sparse_txt
```
Target: 18/18 registered, median relative rotation error < 0.5° vs COLMAP. The reference
solution got 0.19° and AUC@5° = 0.96 with `--five-point`, but only **2/18** with its own
8-point verifier. Reproduce that gap and explain it (flat beach → degenerate F).
*Update from the full implementation:* most of that gap was actually the **12-unknown DLT
PnP** degenerating on the near-planar beach (smallest/largest singular value of the
registered structure is 1–3%), not the F-based verification. With a P3P minimal solver in
`pnp_ransac` (the default now) the 8-point verifier also reaches 18/18 (median rotation
error 0.08° vs COLMAP). Then
implement Nistér's 5-point solver or a homography-aware initialization if you want to
remove the OpenCV dependency.
**Then:** Aukerman (77 images, nadir survey) with `--sequential 10` versus exhaustive matching.
**Stretch:** Refine intrinsics in BA (self-calibration) instead of borrowing COLMAP's.

### M7: Georeferencing and pose evaluation (1–2 sessions)

**Build:** Umeyama Sim3, `transform_reconstruction`, WGS84 → ECEF → ENU, ATE, relative
pose errors, and AUC.
**Gate:** `-k m07`, then georeference your Brighton Beach model: EXIF GPS → ENU → Sim3 from
camera centers. Report GPS residuals in metres. They should be at GPS noise level (1–5 m)
and must not show a systematic trend (a trend means doming or a scale error).
**Stretch:** Use GPS as a soft prior inside BA instead of a post-hoc alignment.

---

## Phase C: Classical multi-view stereo

### M8: Plane-sweep stereo (2–3 sessions)

**Build:** `mvs/plane_sweep.py`: relative pose, plane-induced homography, windowed ZNCC,
and fronto-parallel sweep. Undistort first with `pycolmap.undistort_images`.
**Gate:** `-k m08` (synthetic rooftops), then depth maps for five Brighton Beach views
using COLMAP poses. Pick the depth range from sparse point depths.
**Stretch:** `mvs/patchmatch.py` with slanted planes. Sweep world-horizontal planes for
oblique views. Compare against `pycolmap.patch_match_stereo`, which needs CUDA COLMAP, e.g.
the `colmap/colmap` Docker image.

### M9: Fusion, meshing, geometric evaluation (2 sessions)

**Build:** `mvs/fusion.py` (geometric consistency, fusion) and `eval/geometry.py` (precision,
recall, F-score). Meshing is scaffolded with Open3D Poisson (`--extra mesh`).
**Gate:** `-k m09`, then a fused cloud and a Poisson mesh of Aukerman. Evaluate against a
reference: Brighton Beach ships ODM's `model.laz` (a pseudo-reference, not ground truth).
For true ground truth use ETH3D or UrbanScene3D ([datasets.md](datasets.md)).
**Deliverable:** A CLI `mvs` subcommand you add yourself: model + images → fused PLY + mesh.
(Done: `aerial-recon mvs`, then `aerial-recon eval-geometry` against `model.laz`.)
**Pitfall (Brighton):** COLMAP self-calibrates the FC300S focal ~19% long (2782 px vs
~2340 px from EXIF + sensor width). Reprojection error cannot tell them apart on this
scene, but the reconstructed flying height can: with COLMAP's focal the ground lands 46 m
below the cameras vs the 39.9 m RelativeAltitude, i.e. a ~10 m vertical offset against
ODM's cloud. `sfm --focal-from-exif` fixes the height but, with distortion dropped, adds a
bowl; the real fix is self-calibration (focal + k1, k2) with an EXIF prior.

---

## Phase D: Modern pipeline

### M10: Learned pose estimation (2 sessions)

**Goal:** Measure how feed-forward and learned pose estimators compare with SfM on
aerial data.
**Do:** Run each source and export COLMAP format
([modern/pose_sources.py](../src/aerial_recon/modern/pose_sources.py) lists commands):
GLOMAP (`--mapper global`), VGGT, and MASt3R-SfM. Install them in separate venvs under
`third_party/` to keep this environment clean.
**Gate:** A table from `compare-poses` against COLMAP (AUC@3/5/10°, ATE after Sim3,
runtime, peak VRAM) on Brighton Beach and a 60-frame Aukerman subset.
**8 GB notes:** VGGT at 518 px handles roughly 30–80 frames per pass. For longer sequences,
run overlapping chunks and stitch them with your M7 Sim3. That is a good mini-project.

### M11: 3D Gaussian Splatting on drone imagery (3–4 sessions)

**Build:** `modern/splat.py` on `gsplat.rasterization`. You wrote the rasterizer in
`~/gsplat`; this time the work is data handling, appearance embeddings, pose refinement,
and fitting in memory.
**Setup:** `uv sync --extra splat`. `~/gsplat` needed pip-installed CUDA toolchain pins
because the system `nvcc` is 11.5, so reuse its `cuda` extra.
**Gate:** Brighton Beach at ¼ resolution trained on the shared holdout split
(`eval.images.holdout_split`): test PSNR, SSIM, LPIPS, and renders. Then the same scene from
VGGT poses, with and without pose refinement.

### M12: Geometry from splats (2 sessions, stretch)

Render depth at training views and fuse it with **your** M9 fusion (or TSDF), or train
2DGS. **Gate:** F-score against the same reference as M9, so M13 can compare geometry
from both tracks fairly.

---

## Phase E: Study

### M13: Classical vs modern on aerial scenes (3+ sessions)

The protocol is in [study.md](study.md) and the experiment matrix is in
[`configs/study.yaml`](../configs/study.yaml). **Deliverable:** A README results section
in the style of `~/gsplat`: one table, one figure, and honest caveats.
