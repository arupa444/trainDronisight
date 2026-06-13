import numpy as np
import pytest
from thermal.preprocess import preprocess


def _highlight_clipped_image():
    # >35% of pixels blown to 255 (highlight_clip > 0.35) -> adaptive CLAHE engages.
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    img[:30, :, :] = 255                      # blown highlights (~47%)
    img[30:, :, 0] = np.tile(np.arange(64, dtype=np.uint8), (34, 1))  # some structure
    return img


def test_preprocess_preserves_shape_and_dtype():
    out = preprocess(_highlight_clipped_image())
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_preprocess_engages_on_clipped_image():
    img = _highlight_clipped_image()
    # adaptive CLAHE should modify a clearly highlight-clipped frame
    assert not np.array_equal(preprocess(img), img)


def test_preprocess_is_deterministic():
    img = _highlight_clipped_image()
    assert np.array_equal(preprocess(img), preprocess(img))


def test_preprocess_rejects_non_rgb():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), dtype=np.uint8))
