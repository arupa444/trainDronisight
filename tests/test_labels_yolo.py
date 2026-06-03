import pytest
from shared.labels import Box, to_yolo_line

def test_converts_center_normalized():
    # box 10..110 x, 20..220 y on 100x100? use explicit dims
    b = Box("wire", xmin=0, ymin=0, xmax=50, ymax=100)
    line = to_yolo_line(b, img_w=100, img_h=200, class_names=["wire"])
    cls, xc, yc, w, h = line.split()
    assert cls == "0"
    assert float(xc) == pytest.approx(0.25)   # (0+50)/2 / 100
    assert float(yc) == pytest.approx(0.25)   # (0+100)/2 / 200
    assert float(w) == pytest.approx(0.5)
    assert float(h) == pytest.approx(0.5)

def test_uses_class_index_from_list():
    b = Box("crossarm_stright", 0, 0, 10, 10)
    names = ["wire", "h_insulator", "v_insulator", "crossarm_stright"]
    assert to_yolo_line(b, 100, 100, names).startswith("3 ")

def test_raises_if_class_not_in_list():
    b = Box("pole", 0, 0, 10, 10)
    with pytest.raises(ValueError):
        to_yolo_line(b, 100, 100, ["wire"])
