"""Images larger than the profiled page are scaled down, not left to hang.

A page with more 40 px cells than the encoder budget used to pass validation and then
wait forever (NOTES, gotchas). `fit_to_budget` must bring it within
HYPERINKLING_IMAGE_MAX_CELLS with the aspect ratio kept, and leave smaller ones alone.

    venv/bin/python -m pytest -q tests/test_image_budget.py
"""

import math

import numpy as np
import pytest

from vllm.models.inkling2.processing import PATCH_SIZE, Inkling2ImageProcessor

A4_150 = 31 * 44  # 1240 x 1760 px, start.sh's default profile


def cells(pixels):
    h, w = pixels.shape[:2]
    return math.ceil(h / PATCH_SIZE) * math.ceil(w / PATCH_SIZE)


@pytest.mark.parametrize("size", [(3508, 2480), (2480, 3508), (2000, 2000)])
def test_large_images_fit(monkeypatch, size):
    monkeypatch.setenv("HYPERINKLING_IMAGE_MAX_CELLS", str(A4_150))
    pixels = np.random.default_rng(0).integers(0, 255, (*size, 3), dtype=np.uint8)
    out = Inkling2ImageProcessor.fit_to_budget(pixels)
    assert cells(out) <= A4_150
    assert out.dtype == np.uint8 and out.shape[2] == 3
    # Aspect ratio kept to within one pixel of rounding.
    assert abs(out.shape[0] / out.shape[1] - size[0] / size[1]) < 2 / min(out.shape[:2])
    # Not shrunk more than needed: one more percent would not fit.
    assert cells(out) > 0.9 * A4_150 or min(out.shape[:2]) < PATCH_SIZE


def test_side_over_model_limit(monkeypatch):
    """Few cells but a side over 4096 px (the model's limit): scaled too."""
    monkeypatch.setenv("HYPERINKLING_IMAGE_MAX_CELLS", str(A4_150))
    out = Inkling2ImageProcessor.fit_to_budget(np.zeros((6000, 200, 3), dtype=np.uint8))
    assert max(out.shape[:2]) <= 4096 and cells(out) <= A4_150


def test_small_and_unset_unchanged(monkeypatch):
    pixels = np.zeros((1754, 1240, 3), dtype=np.uint8)
    monkeypatch.setenv("HYPERINKLING_IMAGE_MAX_CELLS", str(A4_150))
    assert Inkling2ImageProcessor.fit_to_budget(pixels) is pixels
    monkeypatch.delenv("HYPERINKLING_IMAGE_MAX_CELLS")
    big = np.zeros((3508, 2480, 3), dtype=np.uint8)
    assert Inkling2ImageProcessor.fit_to_budget(big) is big
