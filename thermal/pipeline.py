import cv2
import numpy as np

from thermal.colormap import ColorToHeat
from thermal.defects import find_hotspots
from thermal.preprocess import preprocess

# Median Lab-distance above which we consider the palette mismatched.
CALIBRATION_DIST_THRESHOLD = 20.0
# Pad fraction around a detected transformer before scanning for hotspots.
# Conductors/connections frequently sit ABOVE the tank box, so a generous pad is
# needed to cover them (validated on real full-frame inference: 0.15 missed an
# overhead hot connection that 0.30 catches).
HOTSPOT_PAD = 0.30


def analyze_image(img_rgb: np.ndarray, transformer_detector,
                  c2h: ColorToHeat, pad: float = HOTSPOT_PAD):
    """Single-stage cascade. Returns (findings, intensity_map, calibration_ok).

    1. CLAHE-preprocess the frame (matching training) and detect transformers.
    2. For each transformer, scan its (padded) region for hot conductors/connections
       and body hotspots via RELATIVE heat on the raw palette — see defects.find_hotspots.
       (Replaces the old learned wire detector, which could not learn thin densely-
       clustered conductors on a small dataset. CV directly targets "where is it hot".)

    `transformer_detector` is anything exposing `detect(img) -> list[Detection]`, fed a
    BGR array (Ultralytics/cv2 convention — the channel order the model trained on).
    """
    proc_bgr = cv2.cvtColor(preprocess(img_rgb), cv2.COLOR_RGB2BGR)
    intensity, dist = c2h.to_intensity(img_rgb)   # heat read on the RAW RGB palette
    calibration_ok = bool(np.median(dist) < CALIBRATION_DIST_THRESHOLD)

    findings = []
    for t in transformer_detector.detect(proc_bgr):
        findings += find_hotspots(intensity, t.bbox, pad=pad)
    return findings, intensity, calibration_ok
