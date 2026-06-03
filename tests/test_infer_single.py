# tests/test_infer_single.py
import numpy as np
from inference.backends import Detection
from inference.infer_pole import detections_to_records

def test_detections_to_records():
    dets = [Detection("pole", 0.9, (1, 2, 3, 4))]
    recs = detections_to_records(dets)
    assert recs == [{"class": "pole", "confidence": 0.9, "box": [1, 2, 3, 4]}]
