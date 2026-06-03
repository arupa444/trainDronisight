import yaml
from pathlib import Path
from shared.labels import Box
from data_prep.emit_yolo import write_label_file, write_data_yaml

def test_write_label_file(tmp_path):
    boxes = [Box("wire", 0, 0, 50, 100), Box("crossarm_stright", 10, 10, 20, 20)]
    out = tmp_path / "img.txt"
    write_label_file(out, boxes, 100, 200, ["wire", "h_insulator", "v_insulator", "crossarm_stright"])
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("3 ")

def test_write_data_yaml(tmp_path):
    p = write_data_yaml(tmp_path, version="clahe", class_names=["wire", "h_insulator"])
    data = yaml.safe_load(Path(p).read_text())
    assert data["names"] == {0: "wire", 1: "h_insulator"}
    assert data["train"].endswith("images/train/clahe")
