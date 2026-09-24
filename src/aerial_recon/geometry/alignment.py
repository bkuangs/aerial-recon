"""M7 — Similarity alignment: putting a reconstruction into a known frame.

Monocular SfM output lives in an arbitrary frame with arbitrary scale. Aligning it to GPS
positions (georeferencing) or to a reference reconstruction (evaluation) is a 7-DoF
similarity problem with a closed-form solution.

Reading: Umeyama "Least-squares estimation of transformation parameters between two point
patterns" (TPAMI 1991); Horn "Closed-form solution of absolute orientation" (1987);
Zhang & Scaramuzza "A Tutorial on Quantitative Trajectory Evaluation" (IROS 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerial_recon.types import Reconstruction


@dataclass(frozen=True)
class Sim3:
    """x' = s R x + t."""

    s: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.s * points @ self.R.T + self.t


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim3 (or SE3 if with_scale=False) with dst ≈ s R src + t.

    src, dst: (N, 3), N >= 3 non-collinear. Remember the reflection fix (det(R) = +1).
    """
    raise NotImplementedError("M7: implement umeyama")


def transform_reconstruction(recon: Reconstruction, sim3: Sim3) -> Reconstruction:
    """Apply a similarity to every point and camera **in place** and return recon.

    Points: X' = s R X + t. Cameras keep their image content, so their centers move like
    points, their orientation rotates by R, and world-to-camera t must be recomputed.
    Derive it — this is the most common georeferencing bug.
    """
    raise NotImplementedError("M7: implement transform_reconstruction")
