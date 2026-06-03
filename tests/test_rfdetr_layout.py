# tests/test_rfdetr_layout.py
import json
from pathlib import Path
from train_rf_detr.layout import build_rfdetr_view

def test_builds_expected_structure(tmp_path):
    # fake our COCO db for one split
    src = tmp_path / "db" / "components"
    (src / "images" / "train" / "clahe").mkdir(parents=True)
    (src / "images" / "train" / "clahe" / "a.jpg").write_bytes(b"x")
    (src / "annotations").mkdir(parents=True)
    (src / "annotations" / "instances_train_clahe.json").write_text(
        json.dumps({"images": [{"id":1,"file_name":"a.jpg","width":2,"height":2}],
                    "annotations": [], "categories": [{"id":0,"name":"wire"}]}))
    out = build_rfdetr_view(src, version="clahe", dest=tmp_path / "rf")
    assert (Path(out) / "train" / "_annotations.coco.json").exists()
    assert (Path(out) / "train" / "a.jpg").exists()


def test_skips_appledouble_sidecars(tmp_path):
    import json
    from pathlib import Path
    from train_rf_detr.layout import build_rfdetr_view
    src = tmp_path / "db" / "components"
    (src / "images" / "train" / "clahe").mkdir(parents=True)
    (src / "images" / "train" / "clahe" / "a.jpg").write_bytes(b"x")
    (src / "images" / "train" / "clahe" / "._a.jpg").write_bytes(b"\x00\x05")  # AppleDouble
    (src / "annotations").mkdir(parents=True)
    (src / "annotations" / "instances_train_clahe.json").write_text(
        json.dumps({"images": [{"id":1,"file_name":"a.jpg","width":2,"height":2}],
                    "annotations": [], "categories": [{"id":0,"name":"wire"}]}))
    out = build_rfdetr_view(src, version="clahe", dest=tmp_path / "rf")
    assert (Path(out) / "train" / "a.jpg").exists()
    assert not (Path(out) / "train" / "._a.jpg").exists()
