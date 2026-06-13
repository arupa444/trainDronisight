"""Collect -> content-hash merge -> capture grouping -> leakage-safe split ->
transformer-anchored crops. Ported/scoped from the trainDronisight data_prep.
"""
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from thermal.data_prep.labels import Annotation, Box, image_content_hash

_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_TS = re.compile(r"DJI_(\d{14})_")


@dataclass
class Sample:
    image: Path
    xml: Path
    source: str


def collect_samples(source_dirs) -> list:
    """Every image that has a sibling .xml across the given dirs (label = sibling,
    fallback to one level up by stem)."""
    out = []
    for d in source_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if img.name.startswith("._") or img.suffix.lower() not in _IMG_EXTS:
                continue
            xml = img.with_suffix(".xml")
            if not xml.exists():
                alt = img.parent.parent / f"{img.stem}.xml"
                xml = alt if alt.exists() else xml
            if xml.exists():
                out.append(Sample(image=img, xml=xml, source=d.name))
    return out


# ---- content-hash merge: collapse byte-identical copies, union their boxes ----
def _iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a.xmin, b.xmin), max(a.ymin, b.ymin)
    ix1, iy1 = min(a.xmax, b.xmax), min(a.ymax, b.ymax)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    aa = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    ab = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    return inter / (aa + ab - inter)


def _dedup_boxes(boxes, iou_thresh=0.8):
    kept = []
    for b in boxes:
        if any(b.name == k.name and _iou(b, k) >= iou_thresh for k in kept):
            continue
        kept.append(b)
    return kept


def merge_by_image_identity(parsed, iou_thresh=0.8):
    """parsed: {image_path: (Sample, Annotation)} -> (merged, stats).

    Group by image content hash; keep one canonical copy per physical image whose
    boxes are the de-duplicated UNION of all copies. A no-op on disjoint captures.
    """
    by_hash = defaultdict(list)
    for path, (s, ann) in parsed.items():
        by_hash[image_content_hash(path)].append((path, s, ann))

    merged = {}
    spanned = collapsed = unioned = dup_removed = 0
    for group in by_hash.values():
        group.sort(key=lambda t: (t[1].source, t[0].name))
        cpath, csample, cann = group[0]
        if len(group) == 1:
            merged[cpath] = (csample, cann)
            continue
        spanned += 1
        collapsed += len(group) - 1
        all_boxes = [b for (_, _, a) in group for b in a.boxes]
        union = _dedup_boxes(all_boxes, iou_thresh)
        dup_removed += len(all_boxes) - len(union)
        unioned += len(union) - len(cann.boxes)
        merged[cpath] = (csample, Annotation(cann.width, cann.height, union))
    stats = {"input_copies": len(parsed), "unique_images": len(merged),
             "images_spanning_multiple_folders": spanned,
             "duplicate_copies_collapsed": collapsed,
             "boxes_added_by_union": unioned, "overlapping_boxes_removed": dup_removed}
    return merged, stats


# ---- capture grouping + leakage-safe grouped split ----
def parse_capture_time(filename: str):
    m = _TS.search(filename)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def assign_groups(filenames, source: str, gap_seconds: int) -> dict:
    """A >gap_seconds jump in DJI capture time starts a new group. Untimed files
    each become their own group (never merged -> cannot leak)."""
    timed = [(fn, parse_capture_time(fn)) for fn in filenames]
    untimed = [(fn, t) for fn, t in timed if t is None]
    timed = sorted([(fn, t) for fn, t in timed if t is not None], key=lambda x: x[1])
    groups, gid, prev = {}, 0, None
    for fn, t in timed:
        if prev is not None and (t - prev).total_seconds() > gap_seconds:
            gid += 1
        groups[fn] = f"{source}:{gid}"
        prev = t
    for fn, _ in untimed:
        gid += 1
        groups[fn] = f"{source}:{gid}"
    return groups


def grouped_split(items, ratios, seed):
    """Split by GROUP (never splitting one), stratified per source so each location
    appears in train. items: dicts with 'group' and 'source'."""
    rng = random.Random(seed)
    members = defaultdict(list)
    for it in items:
        members[it["group"]].append(it)
    groups_by_source = defaultdict(list)
    for g, its in members.items():
        groups_by_source[its[0]["source"]].append(g)

    out = {"train": [], "val": [], "test": []}
    for source, groups in groups_by_source.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        if not n:
            continue
        n_train = min(max(round(n * ratios["train"]), 1), n)
        rem = n - n_train
        n_val = min(round(n * ratios["val"]), rem)
        if n_val == 0 and rem >= 1:   # tiny source: fill val before test
            n_val = 1
        buckets = {"train": groups[:n_train], "val": groups[n_train:n_train + n_val],
                   "test": groups[n_train + n_val:]}
        for split_name, gs in buckets.items():
            for g in gs:
                out[split_name].extend(members[g])
    return out


# ---- transformer-anchored crops for the wire detector ----
def _pad_clip(box, pad_frac, W, H):
    bw, bh = box.xmax - box.xmin, box.ymax - box.ymin
    px, py = int(round(bw * pad_frac)), int(round(bh * pad_frac))
    return (max(0, box.xmin - px), max(0, box.ymin - py),
            min(W, box.xmax + px), min(H, box.ymax + py))


def _visible_frac(box, crop):
    x0, y0, x1, y1 = crop
    iw = max(0, min(box.xmax, x1) - max(box.xmin, x0))
    ih = max(0, min(box.ymax, y1) - max(box.ymin, y0))
    area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
    return (iw * ih) / area if area > 0 else 0.0


def _remap_clip(box, crop):
    x0, y0, x1, y1 = crop
    return Box(box.name, max(box.xmin, x0) - x0, max(box.ymin, y0) - y0,
               min(box.xmax, x1) - x0, min(box.ymax, y1) - y0)


def make_anchor_crops(ann, keep_classes, anchor_classes, pad_frac, min_visible):
    """One crop per anchor (transformer) box + pad; keep `keep_classes` boxes that
    are >= min_visible inside, remapped to crop-local coords. Returns
    [(crop_xyxy, Annotation)]. Falls back to the union of keep-class boxes when no
    anchor is present so the frame is still used at ~component scale."""
    W, H, keep = ann.width, ann.height, set(keep_classes)
    anchors = [b for b in ann.boxes if b.name in set(anchor_classes)]
    if not anchors:
        sub = [b for b in ann.boxes if b.name in keep]
        if not sub:
            return []
        anchors = [Box("_union", min(b.xmin for b in sub), min(b.ymin for b in sub),
                       max(b.xmax for b in sub), max(b.ymax for b in sub))]
    out = []
    for ab in anchors:
        crop = _pad_clip(ab, pad_frac, W, H)
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        if cw <= 1 or ch <= 1:
            continue
        members = [_remap_clip(b, crop) for b in ann.boxes
                   if b.name in keep and _visible_frac(b, crop) >= min_visible]
        members = [b for b in members if b.xmax > b.xmin and b.ymax > b.ymin]
        if members:
            out.append((crop, Annotation(cw, ch, members)))
    return out
