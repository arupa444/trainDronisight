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


def test_pipeline_attaches_condition_when_detector_given(tmp_path):
    # stage 4: each detected component crop gets a 'conditions' list from the condition model
    img = np.zeros((200, 200, 3), np.uint8)
    pole_det = _Fake([Detection("pole", 0.95, (10, 10, 110, 160))])
    above_det = _Fake([Detection("v_insulator", 0.8, (5, 5, 25, 25))])
    cond_det = _Fake([Detection("v_insulator_broken", 0.7, (0, 0, 10, 10))])
    out = run_pipeline(img, pole_det, above_det, _Fake([]), crop_dir=tmp_path,
                       image_name="x.jpg", pole_pad=0.0, condition_detector=cond_det)
    comp = out["poles"][0]["components_above"][0]
    assert comp["conditions"] == [{"class": "v_insulator_broken", "confidence": 0.7,
                                   "box_comp": [0, 0, 10, 10]}]


def test_pipeline_omits_condition_when_no_detector(tmp_path):
    img = np.zeros((200, 200, 3), np.uint8)
    out = run_pipeline(img, _Fake([Detection("pole", 0.9, (10, 10, 110, 160))]),
                       _Fake([Detection("wire", 0.8, (5, 5, 25, 25))]), _Fake([]),
                       crop_dir=tmp_path, image_name="x.jpg", pole_pad=0.0)
    assert "conditions" not in out["poles"][0]["components_above"][0]


def test_build_detector_selects_backend(monkeypatch):
    import inference.pipeline as P
    calls = {}

    def fake_yolo(w, conf, imgsz):
        calls["yolo"] = (w, conf, imgsz)
        return "Y"

    def fake_rf(w, names, conf, resolution):
        calls["rfdetr"] = (w, names, conf, resolution)
        return "R"

    def fake_frcnn(w, names, conf, min_size):
        calls["frcnn"] = (w, names, conf, min_size)
        return "F"

    monkeypatch.setattr(P, "YoloDetector", fake_yolo)
    monkeypatch.setattr(P, "RFDetrDetector", fake_rf)
    monkeypatch.setattr(P, "TorchvisionDetector", fake_frcnn)
    assert P.build_detector("yolo", "y.pt", 0.25, 1280, ["wire"]) == "Y"
    assert calls["yolo"] == ("y.pt", 0.25, 1280)
    assert P.build_detector("rfdetr", "r.pth", 0.3, 1280, ["wire"], resolution=1008) == "R"
    assert calls["rfdetr"] == ("r.pth", ["wire"], 0.3, 1008)
    # frcnn must be served at the TRAINING min_size (default 2000), not torchvision's 800
    assert P.build_detector("frcnn", "f.pt", 0.4, 1280, ["wire"]) == "F"
    assert calls["frcnn"] == ("f.pt", ["wire"], 0.4, 2000)
    assert P.build_detector("frcnn", "f.pt", 0.4, 1280, ["wire"], frcnn_min_size=1333) == "F"
    assert calls["frcnn"] == ("f.pt", ["wire"], 0.4, 1333)


def test_torchvision_detector_defaults_to_training_resize():
    # regression guard: serving FRCNN at torchvision's 800/1333 instead of the trained
    # 2000/3000 collapses small-object detection. The detector default must match training.
    import inspect
    from inference.backends import TorchvisionDetector
    assert inspect.signature(TorchvisionDetector.__init__).parameters["min_size"].default == 2000
