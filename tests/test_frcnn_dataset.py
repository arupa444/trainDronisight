# tests/test_frcnn_dataset.py
from pathlib import Path
import torch
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
