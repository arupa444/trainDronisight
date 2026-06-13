import numpy as np

from thermal.data_prep.labels import normalize_class_name, Box, Annotation, to_yolo_line
from thermal.data_prep.assemble import (merge_by_image_identity, grouped_split,
                                        assign_groups, make_anchor_crops, Sample)


def test_normalize_collapses_aliases():
    assert normalize_class_name("transformer") == "transformer"
    assert normalize_class_name("front_transformer") == "transformer"
    assert normalize_class_name("transform") == "transformer"
    assert normalize_class_name("WIRE") == "wire"
    assert normalize_class_name("pole") is None  # unknown -> dropped


def test_to_yolo_line_normalized_center():
    line = to_yolo_line(Box("wire", 0, 0, 100, 50), 200, 100, ["transformer", "wire"])
    assert line == "1 0.250000 0.250000 0.500000 0.500000"


def test_merge_unions_partial_labels_of_same_image(tmp_path):
    # Two "annotators" labeled DIFFERENT boxes on byte-identical copies of one photo.
    img = tmp_path / "a.jpg"
    img.write_bytes(b"IDENTICAL-IMAGE-BYTES")
    copy = tmp_path / "b.jpg"
    copy.write_bytes(b"IDENTICAL-IMAGE-BYTES")
    s1 = Sample(image=img, xml=img, source="m1")
    s2 = Sample(image=copy, xml=copy, source="m2")
    a1 = Annotation(100, 100, [Box("transformer", 0, 0, 50, 50)])
    a2 = Annotation(100, 100, [Box("wire", 60, 60, 70, 70)])
    merged, stats = merge_by_image_identity({img: (s1, a1), copy: (s2, a2)})
    assert stats["unique_images"] == 1            # collapsed to one physical image
    assert stats["duplicate_copies_collapsed"] == 1
    (s, ann), = merged.values()
    assert {b.name for b in ann.boxes} == {"transformer", "wire"}  # UNION of both


def test_grouped_split_has_no_leakage():
    items = [{"image": f"i{i}", "key": f"k{i}",
              "group": f"src:{i // 3}", "source": "src"} for i in range(30)]
    split = grouped_split(items, {"train": 0.8, "val": 0.15, "test": 0.05}, seed=1337)
    groups = {sp: {it["group"] for it in its} for sp, its in split.items()}
    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()
    assert sum(len(v) for v in split.values()) == 30  # nothing dropped


def test_assign_groups_splits_on_time_gap():
    g = assign_groups(["DJI_20260101120000_0001_T.JPG", "DJI_20260101120030_0002_T.JPG",
                       "DJI_20260101130000_0003_T.JPG"], "src", gap_seconds=60)
    # first two are 30s apart (same group); third is ~1h later (new group)
    assert g["DJI_20260101120000_0001_T.JPG"] == g["DJI_20260101120030_0002_T.JPG"]
    assert g["DJI_20260101130000_0003_T.JPG"] != g["DJI_20260101120000_0001_T.JPG"]


def test_grouped_split_tiny_source_fills_val_before_test():
    # 3 groups, one source: must give train>=1 and val>=1 (val before test), no leakage
    items = [{"image": f"i{i}", "key": f"k{i}", "group": f"src:{i}", "source": "src"}
             for i in range(3)]
    split = grouped_split(items, {"train": 0.8, "val": 0.15, "test": 0.05}, seed=1337)
    assert len(split["train"]) >= 1
    assert len(split["val"]) >= 1
    assert sum(len(v) for v in split.values()) == 3
    groups = {sp: {it["group"] for it in its} for sp, its in split.items()}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])


def test_make_anchor_crops_fallback_when_no_anchor():
    # No transformer anchor present -> falls back to union of wire boxes as the crop
    ann = Annotation(1000, 1000, [
        Box("wire", 100, 100, 150, 160),
        Box("wire", 300, 300, 340, 380),
    ])
    crops = make_anchor_crops(ann, ["wire"], ["transformer"], pad_frac=0.0, min_visible=0.3)
    assert len(crops) == 1
    _, cann = crops[0]
    assert len(cann.boxes) == 2  # both wires kept, remapped into the union crop


def test_make_anchor_crops_drops_partially_visible_wire():
    ann = Annotation(1000, 1000, [
        Box("transformer", 200, 200, 600, 600),
        Box("wire", 580, 300, 700, 360),  # straddles the right edge; mostly outside
    ])
    # mostly-outside wire dropped at high min_visible, kept at low
    assert make_anchor_crops(ann, ["wire"], ["transformer"], 0.0, 0.8) == [] \
        or len(make_anchor_crops(ann, ["wire"], ["transformer"], 0.0, 0.8)[0][1].boxes) == 0
    kept = make_anchor_crops(ann, ["wire"], ["transformer"], 0.0, 0.1)
    assert len(kept) == 1 and len(kept[0][1].boxes) == 1


def test_make_anchor_crops_remaps_wire_into_transformer_crop():
    ann = Annotation(1000, 1000, [
        Box("transformer", 200, 200, 600, 600),   # anchor
        Box("wire", 300, 300, 350, 360),           # inside -> kept, remapped
        Box("wire", 900, 900, 950, 950),           # outside -> dropped
    ])
    crops = make_anchor_crops(ann, ["wire"], ["transformer"], pad_frac=0.0, min_visible=0.3)
    assert len(crops) == 1
    crop_box, cann = crops[0]
    assert crop_box == (200, 200, 600, 600)        # no pad
    assert len(cann.boxes) == 1                    # only the inside wire
    b = cann.boxes[0]
    assert (b.xmin, b.ymin, b.xmax, b.ymax) == (100, 100, 150, 160)  # remapped to crop-local
