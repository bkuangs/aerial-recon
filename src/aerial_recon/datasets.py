"""Dataset registry. Small ODM datasets download automatically; large ones are documented
in docs/datasets.md because they require registration or hundreds of GB."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path("data")


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    images: int
    size: str
    license: str
    notes: str


DATASETS = {
    d.name: d
    for d in [
        Dataset("brighton_beach", "https://github.com/OpenDroneMap/drone_dataset_brighton_beach",
                18, "~80 MB", "BSD-2-Clause",
                "DJI, oblique+nadir; ships ODM dsm.tif + model.laz as a pseudo-reference. "
                "Your inner dev loop."),
        Dataset("aukerman", "https://github.com/OpenDroneMap/odm_data_aukerman",
                77, "~520 MB", "CC0-1.0",
                "Classic nadir survey over a park with buildings. First real SfM gate."),
        Dataset("toledo", "https://github.com/OpenDroneMap/odm_data_toledo",
                87, "~430 MB", "see repo",
                "DJI survey; a second scene for the study."),
        Dataset("pacifica", "https://github.com/OpenDroneMap/odm_data_pacifica",
                12, "~90 MB", "see license.html",
                "Tiny; good for quick failure-mode experiments."),
    ]
}


def download(name: str, root: Path = DATA_ROOT) -> Path:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name}; choose from {sorted(DATASETS)}")
    dest = root / name
    if dest.exists():
        print(f"{dest} already exists")
        return dest
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", DATASETS[name].url, str(dest)], check=True)
    return dest
