import cv2
import numpy as np
from thermal.schema import DefectFinding
from thermal.report import to_json, annotate, heatmap_overlay


def test_to_json_structure():
    findings = [DefectFinding("wire", (1, 2, 3, 4), "Critical", 0.4123)]
    out = to_json(findings)
    assert out == [{
        "component": "wire",
        "bbox": [1, 2, 3, 4],
        "severity": "Critical",
        "relative_delta": 0.4123,
    }]


def test_annotate_keeps_image_shape():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    findings = [DefectFinding("transformer", (5, 5, 40, 40), "Watch", 0.15)]
    out = annotate(img, findings)
    assert out.shape == img.shape
    assert out is not img            # must not mutate input
    # The rectangle must have been drawn: at least one pixel inside the bbox
    # region should be non-zero (the border was painted).
    bbox_region = out[5:41, 5:41]
    assert np.any(bbox_region != 0), "expected rectangle pixels in bbox region"


def test_heatmap_overlay_keeps_shape():
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    intensity = np.full((30, 30), 0.5, dtype=np.float32)
    out = heatmap_overlay(img, intensity)
    assert out.shape == img.shape
    # The blend must have been applied: output must differ from the all-zero
    # input (colormap contribution is non-zero) and must not equal the raw
    # colormap without blending.
    assert np.any(out != img), "expected heatmap overlay to change pixels"
    # Verify the output is a blend, not just the raw colormap (alpha < 1).
    raw_hm = cv2.applyColorMap(
        (np.clip(intensity, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    assert np.any(out != raw_hm), "expected blended output, not raw colormap"
