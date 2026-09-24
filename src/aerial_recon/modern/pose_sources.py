"""M10 — Pose sources for the modern track. Scaffold.

Every pose estimator in this project is run as an external tool and exports a COLMAP
model; this module just loads any of them into a Reconstruction so they are evaluated
identically (eval.poses) and feed the same splat trainer.

  source      how to produce it                                             notes
  ----------  ------------------------------------------------------------  ------------------------
  colmap      `aerial-recon colmap IMAGES --out OUT`                        reference, slow, robust
  glomap      `aerial-recon colmap IMAGES --out OUT --mapper global`        global SfM, much faster
  yours       `aerial-recon sfm IMAGES --out OUT`                           M6
  vggt        facebookresearch/vggt `demo_colmap.py --scene_dir DIR`        feed-forward, ~1B params
  mast3r_sfm  naver/mast3r `demo.py` (sparse GA) -> export COLMAP           pairwise + global align
  gps_only    EXIF GPS + gimbal angles, no vision (a baseline to beat)      M7

VGGT on an 8 GB GPU: bf16 weights ≈ 2.5 GB; expect roughly 30–80 frames at 518 px per
forward pass. Subsample video frames (M2) or run in overlapping chunks and align the
chunks with Sim3 (M7) — that chunk-and-align step is a nice mini-project in itself.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from aerial_recon.io.colmap_text import read_model
from aerial_recon.types import Reconstruction


def load_poses(model_dir: str | Path) -> Reconstruction:
    """Load a COLMAP model directory in either text or binary format."""
    model_dir = Path(model_dir)
    if (model_dir / "images.txt").exists():
        return read_model(model_dir)
    if (model_dir / "images.bin").exists():
        import pycolmap

        with tempfile.TemporaryDirectory() as tmp:
            pycolmap.Reconstruction(model_dir).write_text(tmp)
            return read_model(tmp)
    raise FileNotFoundError(f"no COLMAP model (images.txt / images.bin) in {model_dir}")
