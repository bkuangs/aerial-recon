"""M11 — 3D Gaussian Splatting on drone imagery with the `gsplat` library.

You already derived and implemented 3DGS from scratch in ~/gsplat; this time the renderer
is `gsplat.rasterization` and the work is everything around it that aerial data demands:

  1. Data: load a COLMAP model (modern.pose_sources.load_poses) + undistorted images;
     holdout split with eval.images.holdout_split so MVS and 3DGS share test views.
  2. Baseline: init from SfM points, standard densification (gsplat.strategy.DefaultStrategy),
     L1 + D-SSIM loss, 30k steps. Fit in 8 GB: downscale images, cap Gaussians, use
     `packed=True`, and consider `absgrad=True` (AbsGS) or `MCMCStrategy` with a budget.
  3. Drone-specific:
       * per-image appearance / exposure embedding (auto-exposure and changing sun),
       * camera pose refinement (learnable SE(3) deltas) — essential for learned poses,
       * scale: scene extent from camera centers, not from a unit cube assumption,
       * sky/background: a far-field sphere or background color for oblique views.
  4. Evaluation: PSNR / SSIM / LPIPS on held-out views. With appearance embeddings, fit the
     test-view embedding on the left half of the image and evaluate on the right half.

Reading: Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023); Ye et al. "gsplat" (2024)
docs and examples/simple_trainer.py; Kulhanek et al. "WildGaussians" (NeurIPS 2024);
Lin et al. "VastGaussian" (CVPR 2024) and Liu et al. "CityGaussian" (ECCV 2024) for scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplatConfig:
    model_dir: Path
    images_dir: Path
    out_dir: Path
    downscale: int = 4
    steps: int = 30_000
    max_gaussians: int = 1_500_000
    sh_degree: int = 3
    appearance_embedding: bool = False
    pose_refinement: bool = False
    test_every: int = 8
    seed: int = 0


def train_splats(config: SplatConfig) -> dict[str, float]:
    """Train, save a checkpoint + test renders under out_dir, and return test metrics."""
    raise NotImplementedError("M11: implement train_splats with gsplat.rasterization")
