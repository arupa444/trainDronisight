import json
from shared.labels import Box, Annotation
from data_prep.emit_coco import build_coco

def test_build_coco_structure():
    anns = {
        "a.jpg": Annotation(100, 200, [Box("wire", 0, 0, 50, 100)]),
        "b.jpg": Annotation(100, 100, [Box("crossarm_stright", 10, 10, 30, 40)]),
    }
    coco = build_coco(anns, class_names=["wire", "h_insulator", "v_insulator", "crossarm_stright"])
    assert {c["name"] for c in coco["categories"]} == \
        {"wire", "h_insulator", "v_insulator", "crossarm_stright"}
    assert len(coco["images"]) == 2
    # COCO bbox is [x, y, w, h]
    wire_ann = [a for a in coco["annotations"] if a["category_id"] == 0][0]
    assert wire_ann["bbox"] == [0, 0, 50, 100]

def test_category_ids_match_class_index():
    coco = build_coco({}, ["wire", "h_insulator"])
    cats = {c["name"]: c["id"] for c in coco["categories"]}
    assert cats == {"wire": 0, "h_insulator": 1}
