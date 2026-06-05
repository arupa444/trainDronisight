"""Offline class-balancing oversampling with bbox-aware augmentation.

For the `component_below_1000` TRAIN split, the rare classes (rust 225 ... top_crossarm 637)
are upsampled by emitting augmented copies of source images until each class approaches the
max class count. Two parts:

  * `plan_oversample` (pure, deterministic) -> the list of source-item indices to augment.
  * `augment_pair` -> applies ONE orientation-aware transform to both the orig and clahe
    image variants (so they stay in sync) and to the boxes, via albumentations ReplayCompose.

Orientation prior: horizontal flip ok, NO vertical flip, only mild rotation (matches
shared/train_args.py). albumentations is imported lazily so `plan_oversample` needs no deps.
"""
import random
from collections import Counter

from shared.labels import Box


def _counts(items, class_names):
    c = Counter()
    for it in items:
        for name in it["classes"]:
            if name in class_names:
                c[name] += 1
    return {cls: c.get(cls, 0) for cls in class_names}


def plan_oversample(items, class_names, seed, target=None, max_factor=20):
    """Greedy plan: repeatedly augment a seeded random image containing the largest-deficit
    class until every class reaches `target` (default = current max class count).

    items: list of dicts each with a 'classes' list. Returns a list of indices into `items`
    (one per augmented copy to create); empty if already balanced or no classes present."""
    counts = _counts(items, class_names)
    if not counts or max(counts.values()) == 0:
        return []
    target = max(counts.values()) if target is None else target
    by_class = {cls: [i for i, it in enumerate(items) if cls in it["classes"]]
                for cls in class_names}
    rng = random.Random(seed)
    running = dict(counts)
    jobs, cap = [], max_factor * max(len(items), 1)
    while len(jobs) < cap:
        deficits = sorted(((target - running[c], c) for c in class_names
                           if running[c] < target and by_class[c]), reverse=True)
        if not deficits:
            break
        cls = deficits[0][1]
        idx = rng.choice(by_class[cls])
        jobs.append(idx)
        for name in items[idx]["classes"]:
            if name in class_names:
                running[name] += 1
    return jobs


def _build_transform(seed):
    import albumentations as A
    # `seed` makes the transform deterministic (albumentations 2.x).
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),                       # left-right ok
            A.Affine(rotate=(-10, 10), scale=(0.85, 1.15),
                     translate_percent=(0.0, 0.06), p=0.9),  # mild only; preserves boxes
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],
                                 min_visibility=0.2),
        seed=seed,
    )


def augment_image(image, boxes, seed_n):
    """Apply one orientation-aware transform to a BGR image + its boxes.
    Returns (aug_image, [Box,...]); deterministic for a given seed_n. Boxes pushed out of
    frame are dropped. The caller derives the clahe variant from aug_image (CLAHE last)."""
    bxs = [[b.xmin, b.ymin, b.xmax, b.ymax] for b in boxes]
    labels = [b.name for b in boxes]
    out = _build_transform(seed_n)(image=image, bboxes=bxs, labels=labels)

    h, w = out["image"].shape[:2]
    new_boxes = []
    for (x1, y1, x2, y2), name in zip(out["bboxes"], out["labels"]):
        xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
        xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
        if xi2 > xi1 and yi2 > yi1:
            new_boxes.append(Box(name, xi1, yi1, xi2, yi2))
    return out["image"], new_boxes
