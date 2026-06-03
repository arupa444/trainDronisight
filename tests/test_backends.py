# tests/test_backends.py
from inference.backends import Detection, parse_yolo_result, filter_detections

class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [xyxy]; self.conf = [conf]; self.cls = [cls]

class _FakeResult:
    def __init__(self, boxes, names): self.boxes = boxes; self.names = names

def test_parse_yolo_result_maps_names():
    res = _FakeResult([_FakeBox([0, 0, 10, 20], 0.9, 0)], {0: "pole"})
    dets = parse_yolo_result(res)
    assert dets == [Detection("pole", 0.9, (0, 0, 10, 20))]

def test_filter_by_confidence():
    dets = [Detection("wire", 0.2, (0, 0, 1, 1)), Detection("wire", 0.8, (0, 0, 1, 1))]
    assert filter_detections(dets, 0.5) == [Detection("wire", 0.8, (0, 0, 1, 1))]
