"""Faster R-CNN training. Usage:
    python -m train_faster_rcnn.train --subset components --version clahe --epochs 30 --batch 2
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shared import config
from shared.device import select_device
from train_faster_rcnn.dataset import CocoDetectionDataset
from train_faster_rcnn.model import build_fasterrcnn


def _collate(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    model.to(device)
    total = 0.0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def run(subset, version, epochs, batch):
    device = select_device()
    class_names = config.SUBSET_CLASSES[subset]
    img_dir = config.COCO_DB / subset / "images" / "train" / version
    ann = config.COCO_DB / subset / "annotations" / f"instances_train_{version}.json"
    ds = CocoDetectionDataset(img_dir, ann)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=_collate)
    model = build_fasterrcnn(num_classes=len(class_names))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.005, momentum=0.9, weight_decay=5e-4)
    out_dir = Path(f"runs/{subset}/faster_rcnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        loss = train_one_epoch(model, dl, opt, device)
        print(f"epoch {ep+1}/{epochs} loss={loss:.4f}")
        torch.save(model.state_dict(), out_dir / "last.pt")
    return out_dir / "last.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=config.SUBSETS, required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.batch)


if __name__ == "__main__":
    main()
