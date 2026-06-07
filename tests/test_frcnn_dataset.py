# tests/test_frcnn_dataset.py
import random
from pathlib import Path
import torch
from PIL import Image
from train_faster_rcnn.dataset import CocoDetectionDataset

ROOT = Path(__file__).parent / "fixtures" / "coco_tiny"

def test_len_and_item():
    ds = CocoDetectionDataset(ROOT / "images", ROOT / "instances.json")
    assert len(ds) == 1
    img, target = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape[0] == 3
    # torchvision FRCNN expects xyxy boxes and 1-based labels (0=background)
    assert target["boxes"].tolist() == [[1, 2, 11, 7]]
    assert target["labels"].tolist() == [1]

def test_augment_hflip_remaps_boxes(monkeypatch):
    # force a horizontal flip and verify x-coords are remapped (x1'=W-x2, x2'=W-x1),
    # boxes stay valid + in-bounds, labels preserved.
    ds = CocoDetectionDataset(ROOT / "images", ROOT / "instances.json", augment=True)
    W = Image.open(ROOT / "images" / "a.jpg").width
    monkeypatch.setattr(random, "random", lambda: 0.0)  # 0.0 < 0.5 -> flip happens
    _, target = ds[0]
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    assert [x1, y1, x2, y2] == [W - 11, 2, W - 1, 7]
    assert 0 <= x1 < x2 <= W
    assert target["labels"].tolist() == [1]
