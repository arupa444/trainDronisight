"""Crop-aligned training data: cut each full frame down to the SCALE the detector is run
on at inference, and remap the boxes into the crop.

Why: the component detectors (above/below) run on the POLE crop and the condition detector
runs on the COMPONENT crop, but they were trained on full ~4000x3000 frames -> a train/serve
spatial-scale gap that hurts thin wires and small insulators. These helpers produce the crop
regions + remapped annotations so a `<subset>_crop` build matches the inference distribution.

Two modes (see config.CROP_ALIGN):
  * "anchor": crop to each anchor box (the pole) + pad; keep in-subset boxes that are at least
    `min_visible` visible inside the crop (clipped to the crop). One crop per pole.
  * "self":   crop to each in-subset box + pad (the component itself, with a little context).
    One crop per component object — matches the per-component crop the condition stage receives.
CLAHE parity note: the build applies CLAHE on the FULL frame and then slices (exactly what
inference does), so these helpers only deal with geometry, never pixels.
"""
from shared.labels import Box, Annotation


def _pad_clip(box, pad_frac, W, H):
    """Pad a box by pad_frac of its size on each side, clipped to the image -> crop xyxy."""
    bw, bh = box.xmax - box.xmin, box.ymax - box.ymin
    px, py = int(round(bw * pad_frac)), int(round(bh * pad_frac))
    x0 = max(0, box.xmin - px)
    y0 = max(0, box.ymin - py)
    x1 = min(W, box.xmax + px)
    y1 = min(H, box.ymax + py)
    return x0, y0, x1, y1


def _visible_frac(box, crop):
    x0, y0, x1, y1 = crop
    ix0, iy0 = max(box.xmin, x0), max(box.ymin, y0)
    ix1, iy1 = min(box.xmax, x1), min(box.ymax, y1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
    return (iw * ih) / area if area > 0 else 0.0


def _remap_clip(box, crop):
    """Box -> crop-local coords, clipped to the crop bounds."""
    x0, y0, x1, y1 = crop
    return Box(box.name,
               max(box.xmin, x0) - x0, max(box.ymin, y0) - y0,
               min(box.xmax, x1) - x0, min(box.ymax, y1) - y0)


def _emit(crop, members, W, H):
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    if cw <= 1 or ch <= 1:
        return None
    boxes = [_remap_clip(b, crop) for b in members]
    boxes = [b for b in boxes if b.xmax > b.xmin and b.ymax > b.ymin]
    return (crop, Annotation(cw, ch, boxes)) if boxes else None


def make_crops(ann, in_subset_classes, mode, anchor_classes=None, pad_frac=0.05, min_visible=0.3):
    """ann: full-frame Annotation (all canonical boxes, incl. the pole). Returns a list of
    (crop_xyxy, crop_Annotation) where the annotation's boxes are remapped to the crop and only
    in_subset_classes are kept. Empty crops are dropped."""
    W, H = ann.width, ann.height
    in_set = set(in_subset_classes)
    out = []

    if mode == "anchor":
        anchors = [b for b in ann.boxes if b.name in set(anchor_classes or ())]
        if not anchors:
            # no anchor (e.g. a pole-less frame): fall back to the union of in-subset boxes so
            # the image is still used, at roughly component scale rather than full-frame.
            sub = [b for b in ann.boxes if b.name in in_set]
            if not sub:
                return []
            anchors = [Box("_union", min(b.xmin for b in sub), min(b.ymin for b in sub),
                           max(b.xmax for b in sub), max(b.ymax for b in sub))]
        for ab in anchors:
            crop = _pad_clip(ab, pad_frac, W, H)
            members = [b for b in ann.boxes if b.name in in_set and _visible_frac(b, crop) >= min_visible]
            emitted = _emit(crop, members, W, H)
            if emitted:
                out.append(emitted)

    elif mode == "self":
        sub = [b for b in ann.boxes if b.name in in_set]
        for target in sub:
            crop = _pad_clip(target, pad_frac, W, H)
            members = [b for b in sub if _visible_frac(b, crop) >= min_visible]
            emitted = _emit(crop, members, W, H)
            if emitted:
                out.append(emitted)
    else:
        raise ValueError(f"unknown crop mode {mode!r}")

    return out
