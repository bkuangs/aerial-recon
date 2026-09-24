"""Image warping helper shared by plane sweep and PatchMatch. Scaffold.

The only subtlety is the half-pixel shift: our homographies map COLMAP-convention pixel
coordinates (pixel centers at +0.5), while OpenCV indexes pixel centers at integers.
"""

from __future__ import annotations

import cv2
import numpy as np

_S = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])  # cv -> colmap
_S_INV = np.linalg.inv(_S)


def warp_image(
    src: np.ndarray, H_ref_to_src: np.ndarray, out_shape: tuple[int, int]  # noqa: N803
) -> tuple[np.ndarray, np.ndarray]:
    """Resample `src` into the reference view: out(p) = src(H p).

    Returns (warped (h, w) float32, valid (h, w) bool) where valid marks pixels whose
    source location falls inside `src`.
    """
    h, w = out_shape
    M = _S_INV @ H_ref_to_src @ _S  # noqa: N806
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped = cv2.warpPerspective(src.astype(np.float32), M, (w, h), flags=flags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    ones = np.ones(src.shape[:2], dtype=np.float32)
    valid = cv2.warpPerspective(ones, M, (w, h), flags=flags,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) > 0.999
    return warped, valid
