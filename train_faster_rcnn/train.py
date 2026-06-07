"""Faster R-CNN training. Usage:
    python -m train_faster_rcnn.train --subset component_above_1000 --version clahe --epochs 30 --batch 8
"""
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def train_one_epoch(model, loader, optimizer, device, desc="train"):
    model.train()
    model.to(device)
    total = 0.0
    # tqdm shows live steps/sec, ETA and running loss (prints periodically in Colab's
    # non-tty !python output, so you can see it's alive mid-epoch).
    pbar = tqdm(loader, desc=desc, mininterval=5.0, dynamic_ncols=True, leave=False)
    for i, (images, targets) in enumerate(pbar, 1):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        pbar.set_postfix(loss=f"{total / i:.4f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_loss(model, loader, device, desc="val"):
    """Validation loss for overfit tracking. torchvision detectors only return losses in
    train() mode, so we forward in train() under no_grad — but put BatchNorm in eval() so
    the val data doesn't pollute the running stats. No backprop."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    total = 0.0
    pbar = tqdm(loader, desc=desc, mininterval=5.0, dynamic_ncols=True, leave=False)
    for i, (images, targets) in enumerate(pbar, 1):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total += float(sum(model(images, targets).values()).item())
        pbar.set_postfix(loss=f"{total / i:.4f}")
    return total / max(len(loader), 1)


def _loader(subset, split, version, batch, workers, shuffle, persistent=True, augment=False):
    img_dir = config.COCO_DB / subset / "images" / split / version
    ann = config.COCO_DB / subset / "annotations" / f"instances_{split}_{version}.json"
    if not Path(ann).exists() or not Path(img_dir).is_dir():
        return None
    ds = CocoDetectionDataset(img_dir, ann, augment=augment)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=_collate,
                      num_workers=workers, pin_memory=(select_device() == "cuda"),
                      persistent_workers=(persistent and workers > 0))


def run(subset, version, epochs, batch, workers=8, patience=7, lr=0.005, min_size=2000):
    device = select_device()
    class_names = config.SUBSET_CLASSES[subset]
    max_size = round(min_size * 1.5)  # fits a 4:3 drone frame's long side at this min_size
    # train loader is AUGMENTED (hflip + color jitter) to fight overfitting on small data;
    # val loader is clean. Parallel prefetch keeps the GPU fed off a slow Drive mount.
    dl = _loader(subset, "train", version, batch, workers, shuffle=True, augment=True)
    # val loader: fewer, non-persistent workers (runs briefly once/epoch) -> avoids 2x pools
    val_dl = _loader(subset, "val", version, batch, min(workers, 4), shuffle=False,
                     persistent=False, augment=False)  # None if the val split isn't present
    model = build_fasterrcnn(num_classes=len(class_names), min_size=min_size, max_size=max_size)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # smooth LR decay
    out_dir = Path(f"runs/{subset}/faster_rcnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    csv_path.write_text("epoch,train_loss,val_loss,lr\n")  # YOLO-style log for plotting/overfit

    # ---- run config banner (printed before training so you can confirm the setup) ----
    n_train = len(dl.dataset)
    n_val = len(val_dl.dataset) if val_dl is not None else 0
    pin = (device == "cuda")
    print("=" * 64)
    print(f"[frcnn] subset={subset}  version={version}  device={device}")
    print(f"[frcnn] classes({len(class_names)}): {', '.join(class_names)}")
    print(f"[frcnn] images: train={n_train} (augmented: hflip+jitter)  |  "
          f"val={n_val} ({'clean' if val_dl is not None else 'MISSING -> val_loss=nan'})")
    print(f"[frcnn] batch={batch}  workers={workers} (val={min(workers, 4)})  "
          f"pin_memory={pin}  epochs={epochs}  patience={patience}  lr={lr}")
    print(f"[frcnn] input resize: min_size={min_size} max_size={max_size} "
          f"(default is 800/1333 — raised for thin wires/small objects)")
    print(f"[frcnn] optimizer=SGD+CosineLR  weight_decay=5e-4  |  outputs -> {out_dir}")
    print("=" * 64, flush=True)

    best_val, bad = float("inf"), 0
    for ep in range(epochs):
        tr = train_one_epoch(model, dl, opt, device, desc=f"epoch {ep+1}/{epochs} train")
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        vl = (eval_loss(model, val_dl, device, desc=f"epoch {ep+1}/{epochs} val")
              if val_dl is not None else float("nan"))
        print(f"epoch {ep+1}/{epochs} train_loss={tr:.4f} val_loss={vl:.4f} lr={cur_lr:.5f}")
        with csv_path.open("a") as f:
            f.write(f"{ep+1},{tr:.6f},{vl:.6f},{cur_lr:.6f}\n")
        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_dl is not None:
            if vl < best_val:                              # new best -> save + reset patience
                best_val, bad = vl, 0
                torch.save(model.state_dict(), out_dir / "best.pt")
            else:
                bad += 1
                if bad >= patience:                        # early stop at the val minimum
                    print(f"early stop @ epoch {ep+1}: val_loss hasn't beaten "
                          f"{best_val:.4f} in {patience} epochs (best.pt kept).")
                    break
    return out_dir / "last.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=config.SUBSETS, required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2),
                    help="DataLoader workers (default = CPU cores, capped at 8); "
                         "raise to hide Drive-mount read latency")
    ap.add_argument("--patience", type=int, default=7,
                    help="early-stop after this many epochs with no val_loss improvement")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--min-size", type=int, default=2000, dest="min_size",
                    help="shorter-side resize (default 2000 high-res; lower if OOM, raise toward "
                         "native ~3000 for max detail)")
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.batch, a.workers, a.patience, a.lr, a.min_size)


if __name__ == "__main__":
    main()
