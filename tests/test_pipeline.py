# tests/test_pipeline.py
import numpy as np
from inference.backends import Detection
from inference.pipeline import run_pipeline

class _Fake:
    def __init__(self, dets): self._dets = dets
    def predict(self, image): return self._dets

def test_pipeline_structures_poles_and_components(tmp_path):
    img = np.zeros((200, 200, 3), np.uint8)
    pole_det = _Fake([Detection("pole", 0.95, (10, 10, 110, 160))])
    # component boxes are in CROP coords (relative to the 100x150 pole crop)
    above_det = _Fake([Detection("wire", 0.8, (5, 5, 25, 25))])
    below_det = _Fake([Detection("vegetation", 0.6, (0, 0, 10, 10))])
    # Production default is pole_pad=0.05; pin 0.0 here to keep the offset arithmetic exact.
    out = run_pipeline(img, pole_det, above_det, below_det,
                       crop_dir=tmp_path, image_name="x.jpg", pole_pad=0.0)
    assert len(out["poles"]) == 1
    pole = out["poles"][0]
    assert pole["confidence"] == 0.95
    # above_1000 detector ran on the crop; box remapped by the pole offset (10,10)
    comp = pole["components_above"][0]
    assert comp["class"] == "wire"
    assert comp["box_full"] == [15, 15, 35, 35]
    assert comp["crop_path"].endswith(".jpg")
    # below_1000 detector also ran on the crop
    veg = pole["components_below"][0]
    assert veg["class"] == "vegetation"
    assert veg["box_full"] == [10, 10, 20, 20]

def test_pipeline_handles_no_poles():
    img = np.zeros((50, 50, 3), np.uint8)
    out = run_pipeline(img, _Fake([]), _Fake([]), _Fake([]), crop_dir=None, image_name="n.jpg")
    assert out["poles"] == []
