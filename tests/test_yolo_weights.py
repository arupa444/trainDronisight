# tests/test_yolo_weights.py
from unittest import mock
from train_yolo import weights

def test_uses_preferred_when_loadable():
    with mock.patch.object(weights, "_loadable", return_value=True):
        name, fell_back = weights.resolve_weights("yolo26x.pt", "yolo11x.pt")
    assert name == "yolo26x.pt" and fell_back is False

def test_falls_back_when_preferred_unavailable():
    with mock.patch.object(weights, "_loadable", side_effect=[False, True]):
        name, fell_back = weights.resolve_weights("yolo26x.pt", "yolo11x.pt")
    assert name == "yolo11x.pt" and fell_back is True

def test_raises_if_neither_loadable():
    with mock.patch.object(weights, "_loadable", return_value=False):
        try:
            weights.resolve_weights("yolo26x.pt", "yolo11x.pt")
            assert False
        except RuntimeError:
            pass
