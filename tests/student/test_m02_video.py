"""M2 gate: frame scoring and keyframe selection. Run: uv run pytest tests/student -k m02"""

import cv2
import numpy as np
import pytest
from scipy import ndimage

from aerial_recon.video.frames import frame_displacement, select_keyframes, sharpness


def textured(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.random((h, w)), 2.0)
    return ((img - img.min()) / np.ptp(img)).astype(np.float32)


def test_sharpness_decreases_with_blur():
    img = textured()
    scores = [sharpness(ndimage.gaussian_filter(img, s)) for s in (0.0, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(scores, scores[1:], strict=False))


def test_sharpness_invariances():
    img = textured()
    assert sharpness(img + 0.2) == pytest.approx(sharpness(img), rel=1e-5)
    assert sharpness(np.full((50, 50), 0.4, dtype=np.float32)) == pytest.approx(0.0, abs=1e-12)


def test_frame_displacement_recovers_translation():
    img = textured(seed=1)
    M = np.float32([[1, 0, 5.0], [0, 1, 3.0]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    assert frame_displacement(img, shifted) == pytest.approx(np.hypot(5.0, 3.0), abs=0.3)
    assert frame_displacement(img, img) == pytest.approx(0.0, abs=0.05)


def test_select_keyframes_uniform_motion():
    n = 31
    keys = select_keyframes(np.ones(n), np.ones(n - 1), min_displacement=10.0, search_window=1)
    assert keys == [0, 10, 20, 30]


def test_select_keyframes_skips_blurry_frames():
    n = 31
    sharp = np.ones(n)
    sharp[1] = 2.0  # sharpest frame in the first window -> first keyframe
    sharp[11] = 0.1  # motion-blurred frame exactly where the threshold is reached
    keys = select_keyframes(sharp, np.ones(n - 1), min_displacement=10.0, search_window=3)
    assert keys[0] == 1
    assert keys[1] == 12


def test_select_keyframes_hover_produces_no_new_keyframes():
    n = 50
    disp = np.zeros(n - 1)
    disp[40:] = 3.0  # drone hovers, then moves
    keys = select_keyframes(np.ones(n), disp, min_displacement=9.0, search_window=1)
    assert keys == [0, 43, 46, 49]
