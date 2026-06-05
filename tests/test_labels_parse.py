from pathlib import Path
from shared.labels import parse_voc

FIX = Path(__file__).parent / "fixtures" / "sample.xml"

def test_parses_size():
    ann = parse_voc(FIX)
    assert (ann.width, ann.height) == (4096, 3072)

def test_keeps_only_kept_classes_and_normalizes():
    ann = parse_voc(FIX)
    names = [b.name for b in ann.boxes]
    assert names == ["pole", "crossarm_stright", "rust"]  # rust now kept (component_below_1000)

def test_box_coords_are_ints():
    ann = parse_voc(FIX)
    pole = ann.boxes[0]
    assert (pole.xmin, pole.ymin, pole.xmax, pole.ymax) == (10, 20, 110, 220)
