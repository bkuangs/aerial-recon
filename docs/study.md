# Study design (M13): classical vs modern on aerial scenes

## Question

> On drone imagery, where does a classical SfM + MVS mesh beat 3D Gaussian Splatting, and
> the reverse? How much of the difference comes from the **pose source** rather than the
> **scene representation**?

Keep one question. Anything else goes in `docs/ideas/`.

## Design

A 2 × 2 core matrix, with one optional arm per axis:

| | MVS mesh (M8–M9) | 3DGS (M11) |
|---|---|---|
| **COLMAP poses** | A | B |
| **Learned poses** (VGGT or MASt3R-SfM) | C | D (± pose refinement) |

Optional: your own SfM as a third pose source, and 2DGS/splat-depth fusion (M12) as a third
representation.

Scenes: **Brighton Beach** (small and flat), **Aukerman** (nadir survey), and one scene
with real geometric ground truth, either an UrbanScene3D scene or a MatrixCity block.
Three seeds per stochastic cell (RANSAC, 3DGS).

## Metrics

| Axis | Metric | Notes |
|------|--------|-------|
| Poses | AUC@3/5/10°, ATE (m, after Sim3 to GT or GPS) | `aerial-recon compare-poses` |
| Geometry | precision / recall / F-score at τ, accuracy, completeness | sample points from meshes and splat surfaces; align to GT first |
| Appearance | PSNR / SSIM / LPIPS on the shared holdout (`holdout_split`, every 8th) | the mesh is rendered with vertex colors from the test cameras |
| Cost | wall time per stage, peak VRAM / RAM | log with `torch.cuda.max_memory_allocated` and `/usr/bin/time -v` |

Pick τ per scene from the ground-truth resolution, e.g. 5, 10, and 20 cm for aerial
scenes. Report all three thresholds.

## Hypotheses (write your predictions *before* running)

* **H1:** 3DGS beats the textured mesh on held-out PSNR/LPIPS in every scene by a large
  margin.
* **H2:** The MVS mesh beats raw 3DGS geometry on F-score. 2DGS or depth fusion (M12)
  closes most of the gap.
* **H3:** Learned poses match COLMAP on Brighton Beach (few images, low texture) but fall
  behind on the Aukerman survey (many images, long baselines). Pose refinement during 3DGS
  recovers most of the appearance gap, but none of the geometry gap in cell C.

## Protocol

1. Freeze the image sets (keyframes and downscale factors) and the holdout split per scene.
   Commit the lists.
2. Produce every pose source, then evaluate poses before any dense work.
3. Run the dense cells from `configs/study.yaml`. Write every metric to
   `outputs/study/<scene>/<cell>/metrics.json`.
4. Present one table (mean ± std over seeds) and one figure (render and error-map crops
   at matched views), then list the failure cases honestly.

## Threats to validity

* A COLMAP-based reference makes COLMAP-pose cells look better. Use independent ground
  truth (LiDAR or synthetic) for any pose or geometry claim.
* Appearance embeddings can overfit test views. Fit test embeddings on half of each test
  image and evaluate on the other half.
* The 8 GB limit forces downscaling. Keep the resolution identical across cells in a scene.
