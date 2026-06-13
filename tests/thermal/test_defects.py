import math

import numpy as np

from thermal.defects import severity_from_delta, find_hotspots, _pad_box


def test_severity_buckets():
    # recall-tuned floors (0.38/0.48/0.58): typical warm connection (<0.38) doesn't alarm
    assert severity_from_delta(0.30) == "Normal"
    assert severity_from_delta(0.42) == "Watch"
    assert severity_from_delta(0.52) == "Investigate"
    assert severity_from_delta(0.65) == "Critical"


def test_severity_nonfinite_is_normal_not_critical():
    # a non-finite delta (NaN/inf) must never silently become Critical
    assert severity_from_delta(float("nan")) == "Normal"
    assert severity_from_delta(math.inf) == "Normal"


def test_pad_box_normalizes_reversed_and_clips():
    # reversed coords normalized; pad clipped to image bounds
    assert _pad_box((60, 90, 40, 10), 0.0, (100, 100)) == (40, 10, 60, 90)
    assert _pad_box((10, 10, 30, 30), 0.5, (100, 100)) == (0, 0, 40, 40)


def test_find_hotspots_detects_hot_region():
    intensity = np.full((200, 200), 0.30, dtype=np.float32)  # warm body
    intensity[80:100, 80:100] = 0.95                          # a white-hot blob (delta ~0.65)
    findings = find_hotspots(intensity, (0, 0, 200, 200), pad=0.0)
    assert len(findings) >= 1
    hot = findings[0]                                         # sorted hottest-first
    assert hot.severity == "Critical"
    x1, y1, x2, y2 = hot.bbox
    assert 70 <= x1 <= 85 and 70 <= y1 <= 85 and 95 <= x2 <= 110 and 95 <= y2 <= 110
    assert hot.component == "hotspot"


def test_find_hotspots_none_on_uniform_crop():
    intensity = np.full((100, 100), 0.5, dtype=np.float32)
    assert find_hotspots(intensity, (0, 0, 100, 100), pad=0.0) == []


def test_find_hotspots_ignores_cold_background():
    # 60% cold background + 40% warm body, NO hot region -> no false hotspot
    intensity = np.full((100, 100), 0.05, dtype=np.float32)
    intensity[:, 60:] = 0.40
    assert find_hotspots(intensity, (0, 0, 100, 100), pad=0.0) == []
