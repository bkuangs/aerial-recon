"""Reference SfM with COLMAP (via pycolmap). Scaffold, not homework.

Run this first on every dataset. It is your answer key for M6/M7 and the pose source for
the "COLMAP poses" arm of the comparison study. `mapper="global"` runs GLOMAP (global SfM),
which is merged into COLMAP >= 3.13 / pycolmap 4.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from aerial_recon.io.colmap_text import read_model
from aerial_recon.types import Reconstruction


def run_colmap(
    image_dir: str | Path,
    out_dir: str | Path,
    matcher: str = "exhaustive",
    mapper: str = "incremental",
    single_camera: bool = True,
    max_image_size: int = 2000,
    overwrite: bool = False,
) -> Reconstruction:
    """Features -> matching -> mapping. Writes `out_dir/sparse/0` (binary) and
    `out_dir/sparse_txt` (text), and returns the largest model.

    matcher: "exhaustive" (unordered photos, < ~300 images) or "sequential" (video frames).
    """
    import pycolmap

    image_dir, out_dir = Path(image_dir), Path(out_dir)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    database = out_dir / "database.db"
    sparse = out_dir / "sparse"
    sparse.mkdir(exist_ok=True)

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max_image_size
    camera_mode = pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO
    pycolmap.extract_features(database, image_dir, camera_mode=camera_mode,
                              extraction_options=extraction)
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(database)
    elif matcher == "sequential":
        pycolmap.match_sequential(database)
    else:
        raise ValueError(f"unknown matcher {matcher}")

    if mapper == "incremental":
        models = pycolmap.incremental_mapping(database, image_dir, sparse)
    elif mapper == "global":
        models = pycolmap.global_mapping(database, image_dir, sparse)
    else:
        raise ValueError(f"unknown mapper {mapper}")
    if not models:
        raise RuntimeError("COLMAP failed to reconstruct any model")

    best = max(models.values(), key=lambda m: m.num_reg_images())
    best_dir = sparse / "0"
    best_dir.mkdir(exist_ok=True)
    best.write(best_dir)
    txt = out_dir / "sparse_txt"
    txt.mkdir(exist_ok=True)
    best.write_text(txt)
    print(best.summary())
    return read_model(txt)
