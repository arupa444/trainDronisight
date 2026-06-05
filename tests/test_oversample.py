from collections import Counter

import numpy as np

from shared.labels import Box
from data_prep.oversample import plan_oversample, augment_image

CLASSES = ["rust", "top_crossarm"]


def _final_counts(items, jobs):
    c = Counter()
    for it in items:
        c.update(it["classes"])
    for j in jobs:
        c.update(items[j]["classes"])
    return c


def test_plan_equalizes_class_counts():
    items = [{"classes": ["rust"]}] + [{"classes": ["top_crossarm"]} for _ in range(4)]
    jobs = plan_oversample(items, CLASSES, seed=1)
    counts = _final_counts(items, jobs)
    assert counts["rust"] == counts["top_crossarm"] == 4   # rust upsampled 1 -> 4


def test_plan_is_deterministic():
    items = [{"classes": ["rust"]}, {"classes": ["rust"]}] + \
            [{"classes": ["top_crossarm"]} for _ in range(3)]
    assert plan_oversample(items, CLASSES, seed=3) == plan_oversample(items, CLASSES, seed=3)


def test_plan_empty_when_already_balanced():
    items = [{"classes": ["rust"]}, {"classes": ["top_crossarm"]}]
    assert plan_oversample(items, CLASSES, seed=1) == []


def test_augment_image_deterministic_and_in_bounds():
    img = (np.zeros((120, 160, 3)) + 127).astype(np.uint8)
    boxes = [Box("rust", 20, 20, 60, 90)]
    a1, b1 = augment_image(img, boxes, seed_n=7)
    a2, b2 = augment_image(img, boxes, seed_n=7)
    assert a1.shape == img.shape
    assert np.array_equal(a1, a2)
    key = lambda bs: [(b.name, b.xmin, b.ymin, b.xmax, b.ymax) for b in bs]
    assert key(b1) == key(b2)
    for b in b1:
        assert 0 <= b.xmin < b.xmax <= 160 and 0 <= b.ymin < b.ymax <= 120
