"""Merge per-member annotations of the SAME physical image into one complete label.

The 6th-june condition data was annotated by 8-9 members, each assigned DIFFERENT
classes over the SAME image pool. So one physical photo appears in several
`6thMem*AllTeam1` folders, and each copy carries only that member's classes
(partial labels). If we treat each folder copy as an independent training image
(the default source-namespaced keying) we get two failures:

  1. LEAKAGE - copies of one photo scatter across train/val/test, so the model is
     evaluated on a frame it trained on.
  2. PARTIAL-LABEL POISONING - a real object that one member didn't label is
     unlabeled (= background) in that copy, so the detector is taught to suppress it.

`merge_by_image_identity` collapses every copy of a physical image (identified by
its byte content, robust to folder/name) into ONE entry whose boxes are the UNION
of all members' boxes, de-duplicating boxes that more than one member drew for the
same object (same class + high IoU). This is a no-op when folders don't overlap, so
it is always safe to run.
"""
import hashlib
from collections import defaultdict

from shared.labels import Annotation, Box


def image_content_hash(path) -> str:
    """MD5 of the raw image bytes. Same content == same physical image regardless of
    folder or filename, so this is the safe cross-folder identity key (DJI counters
    reset per card, so filename stems alone are not trustworthy across captures)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a.xmin, b.xmin), max(a.ymin, b.ymin)
    ix1, iy1 = min(a.xmax, b.xmax), min(a.ymax, b.ymax)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    return inter / (area_a + area_b - inter)


def _dedup_boxes(boxes, iou_thresh=0.8):
    """Drop boxes that duplicate an already-kept box of the SAME class (IoU >= thresh).
    Two members labeling the same object would otherwise produce overlapping copies."""
    kept = []
    for b in boxes:
        if any(b.name == k.name and _iou(b, k) >= iou_thresh for k in kept):
            continue
        kept.append(b)
    return kept


def merge_by_image_identity(parsed, iou_thresh=0.8):
    """parsed: {image_path: (Sample, Annotation)}.

    Group by image content hash; for each group, keep ONE canonical copy (the path
    whose (source, name) sorts first, deterministic) and replace its annotation with
    the de-duplicated UNION of every copy's boxes. Returns (merged_parsed, stats).
    Copies that are unique (appear in only one folder) pass through unchanged.
    """
    by_hash = defaultdict(list)
    for path, (s, ann) in parsed.items():
        by_hash[image_content_hash(path)].append((path, s, ann))

    merged = {}
    n_groups_merged = 0          # physical images that spanned >1 copy
    n_copies_dropped = 0         # extra folder copies collapsed away
    n_boxes_unioned = 0          # boxes gained on canonical copies vs their own labels
    n_dup_boxes_removed = 0      # overlapping same-class boxes removed during union
    for group in by_hash.values():
        group.sort(key=lambda t: (t[1].source, t[0].name))
        canon_path, canon_sample, canon_ann = group[0]
        if len(group) == 1:
            merged[canon_path] = (canon_sample, canon_ann)
            continue
        n_groups_merged += 1
        n_copies_dropped += len(group) - 1
        all_boxes = [b for (_, _, ann) in group for b in ann.boxes]
        union = _dedup_boxes(all_boxes, iou_thresh)
        n_dup_boxes_removed += len(all_boxes) - len(union)
        n_boxes_unioned += len(union) - len(canon_ann.boxes)
        merged[canon_path] = (
            canon_sample,
            Annotation(canon_ann.width, canon_ann.height, union),
        )
    stats = {
        "input_copies": len(parsed),
        "unique_images": len(merged),
        "images_spanning_multiple_folders": n_groups_merged,
        "duplicate_copies_collapsed": n_copies_dropped,
        "boxes_added_by_union": n_boxes_unioned,
        "overlapping_boxes_removed": n_dup_boxes_removed,
    }
    return merged, stats


def _report(subset):
    """Read-only diagnostic: how much do the source folders for `subset` overlap?"""
    from shared import config
    from shared.labels import parse_voc
    from data_prep.collect import collect_samples

    class_names = config.SUBSET_CLASSES[subset]
    samples = collect_samples(config.SUBSET_SOURCE_DIRS.get(subset, config.SOURCE_DIRS))
    parsed = {}
    for s in samples:
        try:
            ann = parse_voc(s.xml)
        except Exception:
            continue
        if any(b.name in class_names for b in ann.boxes):
            parsed[s.image] = (s, ann)
    print(f"[{subset}] parsed copies with >=1 in-subset class: {len(parsed)}")
    _, stats = merge_by_image_identity(parsed)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["images_spanning_multiple_folders"]:
        print("  -> SAME images span multiple member folders: merge IS needed "
              "(prevents split leakage + partial-label poisoning).")
    else:
        print("  -> folders are disjoint: merge is a no-op, current build was already safe.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Report cross-folder image overlap for a subset.")
    ap.add_argument("--subset", default="component_classification")
    _report(ap.parse_args().subset)
