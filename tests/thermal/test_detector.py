import numpy as np
from thermal.detector import YoloDetector


class _FakeBox:
    def __init__(self, cls_id, xyxy, conf):
        self.cls = [cls_id]
        self.xyxy = [xyxy]
        self.conf = [conf]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    names = {0: "transformer", 1: "wire"}
    def __call__(self, img, conf=0.25):
        return [_FakeResult([
            _FakeBox(0, [1.0, 2.0, 30.0, 40.0], 0.9),
            _FakeBox(1, [5.0, 6.0, 10.0, 50.0], 0.7),
        ])]


def test_detector_translates_yolo_output():
    det = YoloDetector.__new__(YoloDetector)  # skip __init__
    det.model = _FakeYOLO()
    det.names = _FakeYOLO.names
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    dets = det.detect(img)
    assert dets[0].cls == "transformer"
    assert dets[0].bbox == (1, 2, 30, 40)
    assert dets[1].cls == "wire"
    assert abs(dets[1].conf - 0.7) < 1e-6
