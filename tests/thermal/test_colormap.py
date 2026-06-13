import numpy as np
import pytest
from thermal.colormap import build_lut, ColorToHeat


def test_single_color_lut_rejected():
    # n=1 would make intensity = idx/(n-1) divide by zero -> must raise, not emit NaN
    with pytest.raises(ValueError):
        ColorToHeat(build_lut("inferno", n=1))


def test_intensity_monotonic_on_palette_gradient():
    lut = build_lut("inferno", n=256)
    c2h = ColorToHeat(lut)
    # Build a 1x256 image that IS the palette, cold->hot left to right.
    img = (lut * 255).astype(np.uint8).reshape(1, -1, 3)
    intensity, dist = c2h.to_intensity(img)
    vals = intensity[0]
    assert vals[0] < 0.05          # coldest color -> ~0
    assert vals[-1] > 0.95         # hottest color -> ~1
    assert np.all(np.diff(vals) >= -1e-6)   # monotonically increasing


def test_calibration_flags_wrong_palette():
    lut = build_lut("inferno", n=256)
    c2h = ColorToHeat(lut)
    # Pure green is far from any inferno color -> poor match.
    green = np.zeros((4, 4, 3), dtype=np.uint8)
    green[..., 1] = 255
    _, dist = c2h.to_intensity(green)
    assert float(np.median(dist)) > 20.0   # high distance = bad calibration
