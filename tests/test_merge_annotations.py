"""Cross-folder per-member annotation merge (6th-june condition data)."""
from dataclasses import dataclass

from shared.labels import Annotation, Box
from data_prep.merge_annotations import merge_by_image_identity, _iou, _dedup_boxes


@dataclass
class FakeSample:
    source: str


def _write_img(tmp_path, folder, name, content):
    d = tmp_path / folder
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_bytes(content)
    return p


def test_same_image_across_members_is_unioned(tmp_path):
    # one physical photo (identical bytes) in two member folders, each labeling a
    # DIFFERENT class -> one merged entry with BOTH boxes, one copy dropped.
    img1 = _write_img(tmp_path, "6thMem1AllTeam1", "DJI_001.JPG", b"PHOTO-A")
    img2 = _write_img(tmp_path, "6thMem3AllTeam1", "DJI_001.JPG", b"PHOTO-A")
    ann1 = Annotation(100, 100, [Box("v_insulator_normal", 0, 0, 10, 10)])
    ann2 = Annotation(100, 100, [Box("wire_normal", 50, 50, 60, 60)])
    parsed = {img1: (FakeSample("6thMem1AllTeam1"), ann1),
              img2: (FakeSample("6thMem3AllTeam1"), ann2)}

    merged, stats = merge_by_image_identity(parsed)

    assert len(merged) == 1
    (sample, ann), = merged.values()
    assert {b.name for b in ann.boxes} == {"v_insulator_normal", "wire_normal"}
    assert stats["images_spanning_multiple_folders"] == 1
    assert stats["duplicate_copies_collapsed"] == 1
    assert stats["boxes_added_by_union"] == 1


def test_distinct_images_pass_through(tmp_path):
    # different bytes -> different physical images -> nothing merged (no-op)
    a = _write_img(tmp_path, "6thMem1AllTeam1", "DJI_001.JPG", b"PHOTO-A")
    b = _write_img(tmp_path, "6thMem2AllTeam1", "DJI_002.JPG", b"PHOTO-B")
    parsed = {a: (FakeSample("6thMem1AllTeam1"), Annotation(10, 10, [Box("wire_normal", 0, 0, 5, 5)])),
              b: (FakeSample("6thMem2AllTeam1"), Annotation(10, 10, [Box("cross_wire", 0, 0, 5, 5)]))}

    merged, stats = merge_by_image_identity(parsed)

    assert len(merged) == 2
    assert stats["images_spanning_multiple_folders"] == 0
    assert stats["duplicate_copies_collapsed"] == 0


def test_two_members_label_same_object_dedups(tmp_path):
    # same photo, both members drew the SAME object (same class, overlapping) -> 1 box kept
    img1 = _write_img(tmp_path, "6thMem1AllTeam1", "DJI_009.JPG", b"DUP")
    img2 = _write_img(tmp_path, "6thMem4AllTeam1", "DJI_009.JPG", b"DUP")
    ann1 = Annotation(100, 100, [Box("h_insulator_broken", 10, 10, 30, 30)])
    ann2 = Annotation(100, 100, [Box("h_insulator_broken", 11, 11, 31, 31)])  # ~same box
    parsed = {img1: (FakeSample("6thMem1AllTeam1"), ann1),
              img2: (FakeSample("6thMem4AllTeam1"), ann2)}

    merged, stats = merge_by_image_identity(parsed)

    (sample, ann), = merged.values()
    assert len(ann.boxes) == 1
    assert stats["overlapping_boxes_removed"] == 1


def test_iou_and_dedup_helpers():
    a = Box("x", 0, 0, 10, 10)
    b = Box("x", 0, 0, 10, 10)
    assert _iou(a, b) == 1.0
    far = Box("x", 100, 100, 110, 110)
    assert _iou(a, far) == 0.0
    # different class but overlapping -> NOT a duplicate
    other = Box("y", 0, 0, 10, 10)
    assert len(_dedup_boxes([a, other])) == 2


# --- condition-conflict resolution (component_classification) -----------------------------
from data_prep.merge_annotations import resolve_cross_class_conflicts, condition_base_and_normal


def test_condition_base_and_normal():
    assert condition_base_and_normal("v_insulator_normal") == ("v_insulator", True)
    assert condition_base_and_normal("v_insulator_broken") == ("v_insulator", False)
    assert condition_base_and_normal("v_insulator_chip_off") == ("v_insulator", False)
    assert condition_base_and_normal("straight_crossarm_band") == ("straight_crossarm", False)
    assert condition_base_and_normal("wire_normal") == ("wire", True)
    assert condition_base_and_normal("cross_wire") == ("wire", False)   # the wire defect class
    assert condition_base_and_normal("om_crossarm_normal") == ("om_crossarm", True)


def test_base_and_normal_covers_all_condition_classes():
    from shared import config
    for s in config.COND_SUBSETS:
        for name in config.SUBSET_CLASSES[s]:
            base, is_norm = condition_base_and_normal(name)
            assert base and isinstance(is_norm, bool)


def test_defect_beats_normal_on_same_object():
    boxes = [Box("v_insulator_normal", 0, 0, 100, 100),
             Box("v_insulator_broken", 2, 2, 100, 100)]   # IoU ~0.96, same base
    resolved, n_override, n_drop = resolve_cross_class_conflicts(boxes)
    assert [b.name for b in resolved] == ["v_insulator_broken"]
    assert n_override == 1 and n_drop == 0


def test_defect_vs_defect_is_dropped_as_ambiguous():
    boxes = [Box("v_insulator_band", 0, 0, 100, 100),
             Box("v_insulator_chip_off", 2, 2, 100, 100)]
    resolved, n_override, n_drop = resolve_cross_class_conflicts(boxes)
    assert resolved == [] and n_drop == 2 and n_override == 0


def test_different_base_components_are_kept():
    # a v_insulator and an h_insulator overlapping is legit co-location, not a conflict
    boxes = [Box("v_insulator_normal", 0, 0, 100, 100),
             Box("h_insulator_normal", 2, 2, 100, 100)]
    resolved, n_override, n_drop = resolve_cross_class_conflicts(boxes)
    assert sorted(b.name for b in resolved) == ["h_insulator_normal", "v_insulator_normal"]
    assert n_override == 0 and n_drop == 0


def test_non_overlapping_same_class_both_kept():
    boxes = [Box("v_insulator_broken", 0, 0, 10, 10),
             Box("v_insulator_broken", 500, 500, 520, 520)]  # far apart -> two real objects
    resolved, _, _ = resolve_cross_class_conflicts(boxes)
    assert len(resolved) == 2
