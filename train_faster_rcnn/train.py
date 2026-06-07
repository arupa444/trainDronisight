"""Faster R-CNN training. Usage:
    python -m train_faster_rcnn.train --subset component_above_1000 --version clahe --epochs 30 --batch 8
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Route DataLoader-worker IPC through temp files instead of /dev/shm. Colab/Docker give a
# tiny /dev/shm, so with num_workers>0 + large image tensors you otherwise hit
# "unable to allocate shared memory (shm) ... Resource temporarily unavailable".
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

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


@torch.no_grad()
def eval_loss(model, loader, device):
    """Validation loss for overfit tracking. torchvision detectors only return losses in
    train() mode, so we forward in train() under no_grad — but put BatchNorm in eval() so
    the val data doesn't pollute the running stats. No backprop."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    total = 0.0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total += float(sum(model(images, targets).values()).item())
    return total / max(len(loader), 1)


def _loader(subset, split, version, batch, workers, shuffle, persistent=True):
    img_dir = config.COCO_DB / subset / "images" / split / version
    ann = config.COCO_DB / subset / "annotations" / f"instances_{split}_{version}.json"
    if not Path(ann).exists() or not Path(img_dir).is_dir():
        return None
    ds = CocoDetectionDataset(img_dir, ann)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=_collate,
                      num_workers=workers, pin_memory=(select_device() == "cuda"),
                      persistent_workers=(persistent and workers > 0))


def run(subset, version, epochs, batch, workers=8):
    device = select_device()
    class_names = config.SUBSET_CLASSES[subset]
    # Parallel prefetch so the GPU isn't starved waiting on (slow) image reads — matters a
    # lot when training straight off a Google Drive mount.
    dl = _loader(subset, "train", version, batch, workers, shuffle=True)
    # val loader: fewer, non-persistent workers (runs briefly once/epoch) -> avoids 2x pools
    val_dl = _loader(subset, "val", version, batch, min(workers, 4), shuffle=False,
                     persistent=False)  # None if the val split isn't present locally
    model = build_fasterrcnn(num_classes=len(class_names))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.005, momentum=0.9, weight_decay=5e-4)
    out_dir = Path(f"runs/{subset}/faster_rcnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    csv_path.write_text("epoch,train_loss,val_loss\n")  # YOLO-style log for plotting/overfit
    best_val = float("inf")
    for ep in range(epochs):
        tr = train_one_epoch(model, dl, opt, device)
        vl = eval_loss(model, val_dl, device) if val_dl is not None else float("nan")
        print(f"epoch {ep+1}/{epochs} train_loss={tr:.4f} val_loss={vl:.4f}")
        with csv_path.open("a") as f:
            f.write(f"{ep+1},{tr:.6f},{vl:.6f}\n")
        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_dl is not None and vl < best_val:           # keep the best-generalizing weights
            best_val = vl
            torch.save(model.state_dict(), out_dir / "best.pt")
    return out_dir / "last.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=config.SUBSETS, required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers; raise to hide Drive-mount read latency")
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.batch, a.workers)


if __name__ == "__main__":
    main()
