# tests/test_geometry.py
import numpy as np
from inference.backends import Detection
from inference.geometry import crop_with_pad, shift_detection

def test_crop_with_pad_returns_subimage_and_offset():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop, (ox, oy) = crop_with_pad(img, (40, 40, 60, 60), pad_frac=0.0)
    assert crop.shape[:2] == (20, 20)
    assert (ox, oy) == (40, 40)

def test_crop_pad_clamps_to_image_bounds():
    img = np.zeros((100, 100, 3), np.uint8)
    crop, (ox, oy) = crop_with_pad(img, (0, 0, 10, 10), pad_frac=1.0)
    assert ox == 0 and oy == 0          # cannot go negative
    assert crop.shape[0] <= 100

def test_shift_detection_maps_crop_to_full():
    d = Detection("wire", 0.9, (5, 5, 15, 15))
    shifted = shift_detection(d, off_x=40, off_y=40)
    assert shifted.box == (45, 45, 55, 55)
    assert shifted.class_name == "wire" and shifted.confidence == 0.9
