from pathlib import Path
from types import SimpleNamespace

from shared.labels import Box, Annotation
from data_prep.dedup import drop_duplicate_annotations

PAIRS = [("mem7", "mem 7.1 5th june")]


def _entry(source, boxes):
    return (SimpleNamespace(source=source), Annotation(100, 100, boxes))


def test_drops_identical_annotation_from_secondary():
    boxes = [Box("pole", 1, 1, 2, 2)]
    parsed = {
        Path("/d/mem7/DJI_1.jpg"): _entry("mem7", list(boxes)),
        Path("/d/mem 7.1 5th june/DJI_1.jpg"): _entry("mem 7.1 5th june", list(boxes)),
    }
    kept, n = drop_duplicate_annotations(parsed, PAIRS)
    assert n == 1
    assert sorted(s.source for s, _ in kept.values()) == ["mem7"]  # secondary dropped


def test_keeps_both_when_annotation_differs():
    parsed = {
        Path("/d/mem7/DJI_2.jpg"): _entry("mem7", [Box("pole", 1, 1, 2, 2)]),
        Path("/d/mem 7.1 5th june/DJI_2.jpg"): _entry("mem 7.1 5th june", [Box("pole", 1, 1, 3, 3)]),
    }
    kept, n = drop_duplicate_annotations(parsed, PAIRS)
    assert n == 0 and len(kept) == 2


def test_no_pairs_keeps_all():
    parsed = {Path("/d/mem3/DJI_1.jpg"): _entry("mem3", [Box("wire", 1, 1, 2, 2)])}
    kept, n = drop_duplicate_annotations(parsed, [])
    assert n == 0 and len(kept) == 1


def test_unrelated_folder_same_stem_not_dropped():
    # DJI counters reset per card: an identical stem in an unrelated folder is a DIFFERENT
    # image and must never be deduped (dedup is scoped to the configured pair only).
    parsed = {
        Path("/d/mem7/DJI_9.jpg"): _entry("mem7", [Box("pole", 1, 1, 2, 2)]),
        Path("/d/mem3/DJI_9.jpg"): _entry("mem3", [Box("pole", 1, 1, 2, 2)]),
    }
    kept, n = drop_duplicate_annotations(parsed, PAIRS)
    assert n == 0 and len(kept) == 2
