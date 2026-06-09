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


def test_condition_is_mapped_to_component_family(tmp_path):
    # condition model returns BOTH an out-of-family (wire_normal) and an in-family
    # (v_insulator_broken) detection; the pipeline must keep ONLY the in-family one for a
    # v_insulator component, and attach NO condition to a vegetation component.
    img = np.zeros((300, 300, 3), np.uint8)
    pole_det = _Fake([Detection("pole", 0.95, (10, 10, 260, 260))])
    above_det = _Fake([Detection("v_insulator", 0.8, (5, 5, 60, 60))])
    below_det = _Fake([Detection("vegetation", 0.7, (5, 70, 60, 120))])
    cond_det = _Fake([Detection("wire_normal", 0.9, (0, 0, 10, 10)),        # WRONG family -> dropped
                      Detection("v_insulator_broken", 0.7, (1, 1, 20, 20)),  # right family -> kept
                      Detection("straight_crossarm_band", 0.99, (0, 0, 5, 5))])  # wrong family -> dropped
    out = run_pipeline(img, pole_det, above_det, below_det, crop_dir=tmp_path,
                       image_name="x.jpg", pole_pad=0.0, condition_detector=cond_det)
    ins = out["poles"][0]["components_above"][0]
    assert ins["class"] == "v_insulator"
    assert ins["condition"] == {"class": "v_insulator_broken", "confidence": 0.7}  # mapped, not the 0.99 crossarm
    assert [c["class"] for c in ins["conditions"]] == ["v_insulator_broken"]        # out-of-family removed
    veg = out["poles"][0]["components_below"][0]
    assert veg["class"] == "vegetation"
    assert "condition" not in veg and "conditions" not in veg                       # no condition family


def test_result_to_rows_flattens_with_condition():
    from inference.pipeline import result_to_rows, CSV_COLUMNS
    result = {"image": "x.jpg", "poles": [
        {"box": [0, 0, 100, 200], "confidence": 0.9, "crop_path": "p.jpg",
         "components_above": [{"class": "v_insulator", "confidence": 0.8,
                              "box_full": [10, 20, 30, 40], "box_crop": [1, 2, 3, 4],
                              "crop_path": "c.jpg",
                              "condition": {"class": "v_insulator_broken", "confidence": 0.7},
                              "conditions": []}],
         "components_below": []}]}
    rows = result_to_rows(result)
    assert len(rows) == 1
    r = rows[0]
    assert set(r) == set(CSV_COLUMNS)
    assert r["component_class"] == "v_insulator" and r["group"] == "above"
    assert r["condition_class"] == "v_insulator_broken" and r["condition_confidence"] == 0.7
    assert r["comp_x1"] == 10 and r["comp_y2"] == 40


def test_result_to_rows_pole_with_no_components():
    from inference.pipeline import result_to_rows
    result = {"image": "x.jpg", "poles": [
        {"box": [0, 0, 10, 10], "confidence": 0.5, "crop_path": "p.jpg",
         "components_above": [], "components_below": []}]}
    rows = result_to_rows(result)
    assert len(rows) == 1 and rows[0]["component_class"] == "" and rows[0]["pole_confidence"] == 0.5


def test_nms_dedups_same_object_across_detectors(tmp_path):
    # above & below detectors BOTH fire on the SAME crossarm region (high overlap) -> NMS keeps ONE
    # (the higher-confidence above box); a far-away wire is kept.
    img = np.zeros((300, 300, 3), np.uint8)
    pole_det = _Fake([Detection("pole", 0.95, (0, 0, 300, 300))])
    above_det = _Fake([Detection("crossarm_stright", 0.85, (50, 50, 150, 150)),
                       Detection("wire", 0.6, (250, 10, 270, 30))])
    below_det = _Fake([Detection("om_crossarm", 0.7, (52, 52, 152, 152))])   # ~same box as crossarm
    out = run_pipeline(img, pole_det, above_det, below_det, crop_dir=tmp_path,
                       image_name="x.jpg", pole_pad=0.0, nms_iou=0.55)
    classes = ([c["class"] for c in out["poles"][0]["components_above"]]
               + [c["class"] for c in out["poles"][0]["components_below"]])
    assert "om_crossarm" not in classes                 # suppressed (overlapped a higher-conf box)
    assert sorted(classes) == ["crossarm_stright", "wire"]


def test_no_nms_keeps_all(tmp_path):
    img = np.zeros((300, 300, 3), np.uint8)
    pole_det = _Fake([Detection("pole", 0.95, (0, 0, 300, 300))])
    above_det = _Fake([Detection("crossarm_stright", 0.85, (50, 50, 150, 150))])
    below_det = _Fake([Detection("om_crossarm", 0.7, (52, 52, 152, 152))])
    out = run_pipeline(img, pole_det, above_det, below_det, crop_dir=tmp_path,
                       image_name="x.jpg", pole_pad=0.0, nms_iou=1.0)   # disabled
    assert len(out["poles"][0]["components_above"]) == 1
    assert len(out["poles"][0]["components_below"]) == 1


def test_run_basename_file_and_dir(tmp_path):
    from inference.pipeline import run_basename
    assert run_basename("some/path/DJI_0070_D.JPG") == "DJI_0070_D_inference"   # file -> stem
    d = tmp_path / "kml 1"; d.mkdir()
    assert run_basename(str(d)) == "kml 1_inference"                            # dir -> name


def test_unique_run_dir_never_overwrites(tmp_path):
    from inference.pipeline import unique_run_dir
    img = "x/DJI_a.JPG"
    d1 = unique_run_dir(tmp_path, img)
    assert d1.name == "DJI_a_inference"
    d1.mkdir(parents=True)
    d2 = unique_run_dir(tmp_path, img)        # first one now exists -> must differ
    assert d2.name == "DJI_a_inference2" and not d2.exists()
    d2.mkdir()
    d3 = unique_run_dir(tmp_path, img)
    assert d3.name == "DJI_a_inference3"


def test_torchvision_detector_defaults_to_training_resize():
    # regression guard: serving FRCNN at torchvision's 800/1333 instead of the trained
    # 2000/3000 collapses small-object detection. The detector default must match training.
    import inspect
    from inference.backends import TorchvisionDetector
    assert inspect.signature(TorchvisionDetector.__init__).parameters["min_size"].default == 2000
