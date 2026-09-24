# Theory progression

The order is deliberate. Each block is the minimum theory needed for the next milestone,
followed by *checks*: questions you should be able to answer, without notes, before
moving on.

## Core texts (free unless noted)

| Short | Reference |
|-------|-----------|
| **HZ** | Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. (book, not free) |
| **Sz** | Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. — free at szeliski.org/Book |
| **FH** | Furukawa & Hernández, *Multi-View Stereo: A Tutorial* (2015) |
| **Solà** | Solà, Deray, Atchuthan, *A micro Lie theory for state estimation in robotics* (arXiv 1812.01537) |
| **Stachniss** | Cyrill Stachniss' Photogrammetry I/II lectures (YouTube). Aerial-first, highly recommended |

---

## 1. Projective geometry and cameras (M0–M1)

* Homogeneous coordinates, the projective plane, points and lines at infinity (HZ ch. 2).
* The pinhole model, K[R|t], and camera center C = −Rᵀt (HZ ch. 6, Sz §2.1).
* Lens distortion models: radial and Brown–Conrady (Sz §2.1.5; COLMAP camera models docs).
* Rotations: matrices, axis-angle, quaternions, and the exp/log maps on SO(3) (Solà §I–III).

*Checks:* Why does a drone camera's principal point matter less than its k1? What does
the null space of P represent? Derive `so3_log` at θ = π.

## 2. Features and robust estimation (M2–M3)

* Scale space, DoG, SIFT descriptors, and RootSIFT (Lowe 2004; Arandjelović & Zisserman 2012).
* Focus measures and optical flow basics (Sz §9.1, Lucas–Kanade).
* RANSAC and its adaptive stopping rule, plus LO-RANSAC, PROSAC, and MAGSAC as further
  reading (Fischler & Bolles 1981; Raguram et al., USAC, 2013).
* Homographies, DLT, and why normalization matters (HZ ch. 4).

*Checks:* Why does the ratio test work better than an absolute distance threshold? How
many iterations are needed for 50% inliers with sample size 8 at 99.9% confidence?

## 3. Two-view geometry (M4)

* Epipolar geometry, F and E, and their degrees of freedom (HZ ch. 9).
* The normalized 8-point algorithm (Hartley 1997) and the 5-point algorithm (Nistér 2004).
* Degeneracies: planar scenes and pure rotation. Model selection with GRIC (Torr 1998),
  which is how COLMAP chooses between H, F, and E.
* Triangulation: DLT, midpoint, and optimal methods (HZ ch. 12; Lee & Civera 2019).

*Checks:* Why does a flat beach break the 8-point F but not the 5-point E? What happens to
triangulation uncertainty as the baseline-to-depth ratio shrinks, e.g. a drone flying
straight toward its target?

## 4. Nonlinear least squares and bundle adjustment (M5)

* Gauss–Newton, Levenberg–Marquardt, and trust regions (Sz App. A; Nocedal & Wright ch. 10).
* Robust losses and IRLS (Triggs et al. 1999 §3).
* Sparsity, the Schur complement, and gauge freedom (Triggs et al. 1999 §6–9; Agarwal et
  al., *Bundle Adjustment in the Large*, 2010).
* On-manifold optimization: perturb rotations in the tangent space (Solà §IV).

*Checks:* Size and structure of JᵀJ for 100 cameras and 50k points. Which 7 directions are
unobservable? Why is the reduced camera system small enough to solve directly?

## 5. Structure-from-motion systems (M6–M7)

* Incremental SfM: Snavely et al., *Photo Tourism* (2006); Schönberger & Frahm, *SfM
  Revisited* (CVPR 2016), the COLMAP paper.
* Global SfM: rotation averaging and translation averaging, then GLOMAP (Pan et al.,
  ECCV 2024).
* PnP: DLT, P3P, and EPnP (Lepetit et al. 2009).
* Absolute orientation (Umeyama 1991, Horn 1987), geodesy (WGS84, ECEF, ENU), and GNSS
  priors in BA.
* Evaluation: ATE, RPE, and AUC (Zhang & Scaramuzza 2018).
* Aerial specifics: doming from uncorrected radial distortion on nadir grids (James &
  Robson 2014, *Mitigating systematic error in topographic models derived from UAV and
  ground-based image networks*) and rolling shutter.

*Checks:* Why do nadir-only surveys dome, and why do oblique images fix it? What does the
"visibility pyramid" in SfM Revisited buy you?

## 6. Multi-view stereo and surfaces (M8–M9)

* Photo-consistency measures (NCC and ZNCC), plane sweep (Collins 1996; Gallup et al. 2007).
* PatchMatch stereo (Bleyer et al. 2011), Gipuma (Galliani et al. 2015), and COLMAP MVS
  with pixelwise view selection (Schönberger et al., ECCV 2016).
* Depth-map fusion and visibility (Merrell et al. 2007; FH ch. 4).
* Surfaces: Poisson (Kazhdan & Hoppe 2013), Delaunay with graph cuts (Labatut et al. 2009;
  OpenMVS), and TSDF (Curless & Levoy 1996).
* Learned MVS for context: MVSNet (Yao et al. 2018).
* Benchmarks and metrics: DTU, ETH3D (Schöps et al. 2017), and Tanks and Temples
  (Knapitsch et al. 2017).

*Checks:* Why do fronto-parallel sweeps suit nadir imagery? Why do they fail on facades?
What does an F-score at τ = 5 cm hide?

## 7. Differentiable rendering and Gaussian splatting (M11)

You covered this in `~/gsplat`. Refresh, then extend:

* 3DGS (Kerbl et al., SIGGRAPH 2023); gsplat library paper (Ye et al. 2024, arXiv 2409.06765).
* Anti-aliasing and scale: Mip-Splatting (Yu et al., CVPR 2024).
* Densification alternatives: AbsGS, and 3DGS-MCMC (Kheradmand et al., NeurIPS 2024).
* Appearance and in-the-wild data: NeRF-W (Martin-Brualla et al. 2021), WildGaussians
  (Kulhanek et al. 2024).
* Large aerial scenes: Mega-NeRF (Turki et al., CVPR 2022), VastGaussian (Lin et al., CVPR
  2024), CityGaussian (Liu et al., ECCV 2024).

*Checks:* Why does per-image appearance embedding make held-out evaluation tricky? What
limits scene size on 8 GB?

## 8. Learned geometry and pose-free reconstruction (M10, M12)

* DUSt3R (Wang et al., CVPR 2024), MASt3R (Leroy et al., ECCV 2024), MASt3R-SfM
  (Duisterhof et al., 3DV 2025).
* VGGT (Wang et al., CVPR 2025). It predicts cameras, depth, and point maps in a single
  forward pass.
* COLMAP-free splatting: CF-3DGS (Fu et al., CVPR 2024), InstantSplat (Fan et al. 2024).
* Geometry from splats: 2DGS (Huang et al., SIGGRAPH 2024), Gaussian surfels, and
  depth-render-then-fuse.
* Monocular depth priors: Depth Anything V2. You used this for depth guidance in `~/gsplat`.

*Checks:* What does a feed-forward model get wrong on a 300-image nadir survey that
incremental SfM gets right, and vice versa on 20 low-texture images?
