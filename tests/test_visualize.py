import numpy as np
from inference.visualize import render_layers, save_layers, LAYERS


def _result():
    return {"image": "x.jpg", "poles": [{
        "box": [10, 10, 90, 190], "confidence": 0.9, "crop_path": "p.jpg",
        "components": [
            {"class": "v_insulator", "confidence": 0.8,
             "box_full": [20, 30, 60, 70], "box_crop": [0, 0, 40, 40], "crop_path": "c.jpg",
             "condition": {"class": "v_insulator_band", "confidence": 0.6},
             "conditions": [{"class": "v_insulator_band", "confidence": 0.6, "box_comp": [2, 3, 30, 35]}]},
            {"class": "vegetation", "confidence": 0.4,
             "box_full": [0, 100, 99, 199], "box_crop": [0, 0, 99, 99], "crop_path": "v.jpg"}],
    }]}


def test_render_layers_returns_four_views_same_shape():
    frame = np.zeros((200, 100, 3), np.uint8)
    layers = render_layers(frame, _result())
    assert set(layers) == set(LAYERS)
    for img in layers.values():
        assert img.shape == frame.shape
    # drawing happened (layers differ from the blank frame)
    assert (layers["pole"] != frame).any()
    assert (layers["components"] != frame).any()
    assert (layers["conditions"] != frame).any()
    assert (layers["all"] != frame).any()
    # input frame is not mutated
    assert (frame == 0).all()


def test_save_layers_writes_four_files(tmp_path):
    frame = np.zeros((200, 100, 3), np.uint8)
    paths = save_layers(frame, _result(), tmp_path, "img1")
    assert set(paths) == set(LAYERS)
    for name in LAYERS:
        p = tmp_path / name / "img1.jpg"
        assert p.exists()
