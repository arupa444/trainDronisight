import math

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from thermal.schema import DefectFinding

# Severity by how far a hot region sits above the transformer body, in 0..1 intensity
# units. CALIBRATED on 755 real crops: a NORMAL warm conductor/connection sits ~+0.22..+0.44
# above body (p50=0.35), defects form the tail at ~+0.55..+0.65. THESE VALUES TRADE RECALL
# FOR PRECISION (user choice): the floor is 0.38 so a white-hot connection on an already-warm
# body (whose relative margin compresses to ~+0.40) is still caught as Watch — at the cost of
# flagging the top ~30% of warm connections. NOTE: severity here is RELATIVE contrast, not
# absolute danger — the colorized palette is auto-gained per frame so the same physical fault
# grades lower on a hot body than on a cool one. Absolute °C (radiometric data) is the real
# cure; lower _HOTSPOT_MARGIN for more recall, raise it for fewer false alarms.
_WATCH, _INVESTIGATE, _CRITICAL = 0.38, 0.48, 0.58

# A trustworthy warm/background split needs at least this many warm pixels.
_MIN_WARM_FRACTION = 0.05
_MIN_WARM_PIXELS = 16

# Hotspot detection (CV, no learned wire model):
_HOTSPOT_MARGIN = _WATCH            # report a region only if it exceeds body by this (>= Watch)
_MIN_HOTSPOT_AREA_FRAC = 0.0008     # ignore specks (fraction of the crop area)
_HOTSPOT_PERCENTILE = 90            # representative hot level within a blob


def severity_from_delta(delta: float) -> str:
    if not math.isfinite(delta):   # guard NaN/inf -> never silently report Critical
        return "Normal"
    if delta < _WATCH:
        return "Normal"
    if delta < _INVESTIGATE:
        return "Watch"
    if delta < _CRITICAL:
        return "Investigate"
    return "Critical"


def _pad_box(bbox, pad: float, shape) -> tuple[int, int, int, int]:
    """Pad a bbox by `pad` of its size on each side, clipped to the image. Normalizes
    a reversed bbox. The pad lets the crop include conductors/bushings just outside
    the tank box."""
    h, w = shape
    x0, y0, x1, y1 = bbox
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    px, py = int(round((x1 - x0) * pad)), int(round((y1 - y0) * pad))
    return (max(0, int(x0) - px), max(0, int(y0) - py),
            min(w, int(x1) + px), min(h, int(y1) + py))


def _body_reference(flat: np.ndarray) -> float:
    """Median intensity of the warm (equipment) pixels.

    The transformer detection box usually contains cold background (foliage, sky)
    around the unit. We split warm equipment from cold background with an Otsu
    threshold and take the median of the warm side, so background cannot drag the
    reference down and inflate the hotspot delta into a false alarm. Falls back to
    the overall median when the warm cluster is too small to trust (only happens on
    a near-all-background crop, i.e. a bad upstream detection — a real transformer
    crop is warm-body-dominant)."""
    if flat.size == 0:
        return 0.0
    if np.ptp(flat) == 0:  # single value: nothing to split
        return float(np.median(flat))
    try:
        threshold = threshold_otsu(flat)
    except ValueError:
        # not dead code: tiny-range float32 arrays pass the ptp guard but still raise
        return float(np.median(flat))
    warm = flat[flat >= threshold]
    if warm.size < max(_MIN_WARM_PIXELS, int(flat.size * _MIN_WARM_FRACTION)):
        return float(np.median(flat))
    return float(np.median(warm))


def find_hotspots(intensity: np.ndarray, bbox, pad: float = 0.15) -> list[DefectFinding]:
    """Find localized hot regions inside a transformer ROI by RELATIVE heat — no
    learned wire detector needed.

    The bbox (a detected transformer) is padded to include conductors/bushings just
    outside the tank. Within it we estimate the warm-body reference, then take every
    connected region whose intensity exceeds the body by `_HOTSPOT_MARGIN` as a
    `hotspot`, scored by how far above body it sits, sorted hottest-first.

    Note: this is RELATIVE heat. It surfaces a hot conductor/connection that stands
    out from the transformer body (the validated defect signature: a white-hot wire on
    a cooler unit). Two consequences, both inherent to relative-without-absolute-temp:
    a *uniformly* hot tank has no internal contrast to flag; and if the whole unit is
    already very hot (body near the top of the palette), a hot wire's margin compresses
    and may fall below the floor. Lower `_HOTSPOT_MARGIN` for more sensitivity.
    """
    h, w = intensity.shape
    x0, y0, x1, y1 = _pad_box(bbox, pad, (h, w))
    crop = intensity[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    body = _body_reference(crop.ravel())
    mask = (crop >= body + _HOTSPOT_MARGIN).astype(np.uint8)
    if int(mask.sum()) == 0:
        return []
    crop_area = int(crop.shape[0]) * int(crop.shape[1])
    min_area = max(4, int(crop_area * _MIN_HOTSPOT_AREA_FRAC))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    findings = []
    for i in range(1, n_labels):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        sub = crop[cy:cy + ch, cx:cx + cw]
        region = sub[labels[cy:cy + ch, cx:cx + cw] == i]
        delta = float(np.percentile(region, _HOTSPOT_PERCENTILE)) - body
        findings.append(DefectFinding(
            "hotspot", (x0 + cx, y0 + cy, x0 + cx + cw, y0 + cy + ch),
            severity_from_delta(delta), delta))

    findings.sort(key=lambda f: f.relative_delta, reverse=True)  # hottest first
    return findings
