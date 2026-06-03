# tests/test_frcnn_train.py
from pathlib import Path
from train_faster_rcnn.train import train_one_epoch
from train_faster_rcnn.dataset import CocoDetectionDataset
from train_faster_rcnn.model import build_fasterrcnn
from torch.utils.data import DataLoader
import torch

ROOT = Path(__file__).parent / "fixtures" / "coco_tiny"

def test_one_epoch_returns_finite_loss():
    ds = CocoDetectionDataset(ROOT / "images", ROOT / "instances.json")
    dl = DataLoader(ds, batch_size=1, collate_fn=lambda b: tuple(zip(*b)))
    model = build_fasterrcnn(num_classes=4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, dl, opt, device="cpu")
    assert loss == loss and loss >= 0  # finite, non-negative
