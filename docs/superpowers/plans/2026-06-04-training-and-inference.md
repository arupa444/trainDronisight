# Training & Inference Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train three detector families on the DBs from Plan 1 — **YOLO26x** (primary, MPS), **Faster R-CNN** (torchvision), **RF-DETR-L** (Roboflow) — and build a modular two-stage inference pipeline (pole → crop → components → crop → structured JSON) that works behind a backend-agnostic `Detector` interface.

**Architecture:** Each training family is its own package with a thin CLI that reads `shared/config` + `shared/device`. A small, heavily-tested `inference/` layer defines a `Detection` dataclass and a `Detector` protocol with three implementations; pure-function geometry (crop/remap) and pipeline glue are unit-tested with fake detectors, while real training/inference get smoke-run validation gates.

**Tech Stack:** Adds `torch`, `torchvision`, `ultralytics` (YOLO26), `rfdetr`, `pycocotools`, `albumentations` to Plan 1's stack.

**Depends on:** Plan 1 (DBs at `/Volumes/dronisight/{yolo_train_db,RF_DETR_Faster_RCNN_train_db}`).

---

## File Structure

```
trainDronisight/
├── shared/
│   └── train_args.py        # domain-aware YOLO train kwargs (aug policy from spec §6.3/§6.1)
├── train_yolo/
│   ├── __init__.py
│   ├── weights.py           # YOLO26x -> yolo11x fallback resolver
│   ├── train_pole.py        # CLI: train Model 1
│   └── train_components.py  # CLI: train Model 2
├── train_faster_rcnn/
│   ├── __init__.py
│   ├── dataset.py           # CocoDetectionDataset (our COCO db)
│   ├── model.py             # build_fasterrcnn(num_classes)
│   └── train.py             # CLI
├── train_rf_detr/
│   ├── __init__.py
│   ├── layout.py            # build rfdetr-compatible COCO view
│   └── train.py             # CLI
├── inference/
│   ├── __init__.py
│   ├── backends.py          # Detection, Detector protocol, Yolo/Torchvision/RFDetr detectors
│   ├── geometry.py          # crop_with_pad, shift_detection, clamp
│   ├── infer_pole.py        # CLI: Model 1 only
│   ├── infer_components.py  # CLI: Model 2 only
│   └── pipeline.py          # full two-stage chain -> structured JSON
└── tests/
    └── test_*.py
```

**Class indices (from Plan 1, fixed):** pole → `pole=0`; components → `wire=0, h_insulator=1, v_insulator=2, crossarm_stright=3`.

---

### Task 0: Add Plan 2 dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add deps to `pyproject.toml` `dependencies` list**

```toml
    "torch>=2.2",
    "torchvision>=0.17",
    "ultralytics>=8.3",
    "rfdetr>=1.0",
    "pycocotools>=2.0.7",
    "albumentations>=1.4",
```

- [ ] **Step 2: Install**

Run: `source .venv/bin/activate && uv pip install -e ".[dev]"`
Expected: torch/torchvision/ultralytics/rfdetr install succeed (macOS wheels are MPS-capable).

- [ ] **Step 3: Verify torch sees MPS on the M4**

Run: `python -c "import torch; print(torch.backends.mps.is_available())"`
Expected: `True` on the M4 Pro (`False` is fine on the M1/CI — CPU fallback).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add torch/ultralytics/rfdetr deps for training"
```

---

### Task 1: `shared/train_args.py` — domain-aware YOLO augmentation policy

**Files:**
- Create: `shared/train_args.py`
- Test: `tests/test_train_args.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_args.py
from shared.train_args import build_yolo_args

def test_pole_args_basic():
    a = build_yolo_args(subset="pole", data_yaml="/x/data.yaml", device="mps",
                        epochs=100, imgsz=1280, batch=8)
    assert a["data"] == "/x/data.yaml"
    assert a["device"] == "mps"
    assert a["imgsz"] == 1280

def test_orientation_priors_respected():
    a = build_yolo_args("components", "/x.yaml", "cpu", 10, 1280, 4)
    assert a["flipud"] == 0.0          # no vertical flip (poles have up-down prior)
    assert a["degrees"] <= 10          # only mild rotation

def test_components_get_stronger_scale_jitter_for_crop_gap():
    pole = build_yolo_args("pole", "/p.yaml", "cpu", 10, 1280, 4)
    comp = build_yolo_args("components", "/c.yaml", "cpu", 10, 1280, 4)
    assert comp["scale"] > pole["scale"]   # simulate zoomed-in inference crops

def test_close_mosaic_and_seed_set():
    a = build_yolo_args("pole", "/p.yaml", "cpu", 100, 1280, 4)
    assert a["close_mosaic"] >= 10
    assert "seed" in a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_args.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `shared/train_args.py`**

```python
from shared import config


def build_yolo_args(subset, data_yaml, device, epochs, imgsz, batch):
    """Build Ultralytics train kwargs with a domain-aware augmentation policy.

    Spec §6.3: poles/insulators have a strong up-down orientation prior -> no
    vertical flip, only mild rotation, no heavy blur. Spec §6.1: Model 2 is
    trained on full frames but runs on cropped pole regions, so components get a
    wider scale-jitter range to simulate the zoomed-in crop distribution.
    """
    is_components = subset == "components"
    return {
        "data": data_yaml,
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": config.SEED,
        "project": f"runs/{subset}",
        "name": "yolo",
        # augmentation
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,   # outdoor lighting variance
        "fliplr": 0.5,
        "flipud": 0.0,                                  # orientation prior
        "degrees": 10.0,                                # mild only
        "translate": 0.1,
        "scale": 0.9 if is_components else 0.5,         # crop-gap mitigation
        "mosaic": 1.0,
        "close_mosaic": 10,                             # finish on realistic images
        "copy_paste": 0.3 if is_components else 0.0,    # help scarcer component classes
        # schedule
        "cos_lr": True,
        "patience": 30,
        "amp": True,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_args.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add shared/train_args.py tests/test_train_args.py
git commit -m "feat: domain-aware YOLO augmentation policy"
```

---

### Task 2: `train_yolo/weights.py` — YOLO26x → yolo11x fallback

**Files:**
- Create: `train_yolo/__init__.py`, `train_yolo/weights.py`
- Test: `tests/test_yolo_weights.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yolo_weights.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `train_yolo/weights.py`** (and empty `train_yolo/__init__.py`)

```python
import warnings


def _loadable(name: str) -> bool:
    """True if Ultralytics can instantiate/download these weights."""
    try:
        from ultralytics import YOLO
        YOLO(name)
        return True
    except Exception:
        return False


def resolve_weights(preferred: str, fallback: str):
    """Return (weights_name, fell_back_bool). Try preferred (YOLO26x), else fallback."""
    if _loadable(preferred):
        return preferred, False
    warnings.warn(f"{preferred} not loadable on this Ultralytics version; "
                  f"falling back to {fallback}.")
    if _loadable(fallback):
        return fallback, True
    raise RuntimeError(f"Neither {preferred} nor {fallback} could be loaded.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_yolo_weights.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add train_yolo/__init__.py train_yolo/weights.py tests/test_yolo_weights.py
git commit -m "feat: YOLO26x weights resolver with yolo11x fallback"
```

---

### Task 3: `train_yolo/train_pole.py` & `train_components.py` — YOLO CLIs

**Files:**
- Create: `train_yolo/train_pole.py`, `train_yolo/train_components.py`
- Test: `tests/test_yolo_cli.py`

- [ ] **Step 1: Write the failing test (config wiring, not a real train)**

```python
# tests/test_yolo_cli.py
from unittest import mock
from train_yolo import train_pole

def test_train_pole_builds_args_and_calls_yolo():
    fake_model = mock.MagicMock()
    with mock.patch("train_yolo.train_pole.YOLO", return_value=fake_model) as Y, \
         mock.patch("train_yolo.train_pole.resolve_weights", return_value=("yolo26x.pt", False)), \
         mock.patch("train_yolo.train_pole.select_device", return_value="cpu"):
        train_pole.run(version="clahe", epochs=1, imgsz=640, batch=2)
    Y.assert_called_once_with("yolo26x.pt")
    kwargs = fake_model.train.call_args.kwargs
    assert kwargs["device"] == "cpu"
    assert kwargs["data"].endswith("pole/data_clahe.yaml")
    assert kwargs["epochs"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yolo_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `train_yolo/train_pole.py`**

```python
"""Train Model 1 (pole). Usage:
    python -m train_yolo.train_pole --version clahe --epochs 100 --imgsz 1280 --batch 8
"""
import argparse
from ultralytics import YOLO
from shared import config
from shared.device import select_device
from shared.train_args import build_yolo_args
from train_yolo.weights import resolve_weights


def run(version, epochs, imgsz, batch):
    weights, fell_back = resolve_weights("yolo26x.pt", "yolo11x.pt")
    if fell_back:
        print("WARNING: using yolo11x fallback weights.")
    device = select_device()
    data_yaml = str(config.YOLO_DB / "pole" / f"data_{version}.yaml")
    args = build_yolo_args("pole", data_yaml, device, epochs, imgsz, batch)
    model = YOLO(weights)
    return model.train(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    run(a.version, a.epochs, a.imgsz, a.batch)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `train_yolo/train_components.py`** (same shape, `"components"` subset)

```python
"""Train Model 2 (components). Usage:
    python -m train_yolo.train_components --version clahe --epochs 150 --imgsz 1280 --batch 8
"""
import argparse
from ultralytics import YOLO
from shared import config
from shared.device import select_device
from shared.train_args import build_yolo_args
from train_yolo.weights import resolve_weights


def run(version, epochs, imgsz, batch):
    weights, fell_back = resolve_weights("yolo26x.pt", "yolo11x.pt")
    if fell_back:
        print("WARNING: using yolo11x fallback weights.")
    device = select_device()
    data_yaml = str(config.YOLO_DB / "components" / f"data_{version}.yaml")
    args = build_yolo_args("components", data_yaml, device, epochs, imgsz, batch)
    model = YOLO(weights)
    return model.train(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    run(a.version, a.epochs, a.imgsz, a.batch)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_yolo_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add train_yolo/train_pole.py train_yolo/train_components.py tests/test_yolo_cli.py
git commit -m "feat: YOLO26x training CLIs for pole and components"
```

---

### Task 4: `train_faster_rcnn/dataset.py` — COCO dataset

**Files:**
- Create: `train_faster_rcnn/__init__.py`, `train_faster_rcnn/dataset.py`
- Test: `tests/test_frcnn_dataset.py`, `tests/fixtures/coco_tiny/` (1 image + json)

- [ ] **Step 1: Create the fixture**

Run:
```bash
mkdir -p tests/fixtures/coco_tiny/images
python -c "import cv2,numpy as np; cv2.imwrite('tests/fixtures/coco_tiny/images/a.jpg', np.zeros((20,30,3),np.uint8))"
```
Then create `tests/fixtures/coco_tiny/instances.json`:
```json
{"images":[{"id":1,"file_name":"a.jpg","width":30,"height":20}],
 "annotations":[{"id":1,"image_id":1,"category_id":0,"bbox":[1,2,10,5],"area":50,"iscrowd":0}],
 "categories":[{"id":0,"name":"wire"}]}
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_frcnn_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Write `train_faster_rcnn/dataset.py`** (and empty `__init__.py`)

```python
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    """Reads our COCO db. Converts [x,y,w,h] cat_id(0-based) ->
    xyxy boxes + 1-based labels (torchvision reserves 0 for background)."""

    def __init__(self, images_dir, ann_json):
        self.images_dir = Path(images_dir)
        coco = json.loads(Path(ann_json).read_text())
        self.images = {im["id"]: im for im in coco["images"]}
        self.by_image = {}
        for a in coco["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)
        self.ids = sorted(self.images)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        boxes, labels = [], []
        for a in self.by_image.get(img_id, []):
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"] + 1)  # 0 reserved for background
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return to_tensor(img), target
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_frcnn_dataset.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add train_faster_rcnn/__init__.py train_faster_rcnn/dataset.py tests/test_frcnn_dataset.py tests/fixtures/coco_tiny/
git commit -m "feat: torchvision COCO dataset for Faster R-CNN"
```

---

### Task 5: `train_faster_rcnn/model.py` — model builder

**Files:**
- Create: `train_faster_rcnn/model.py`
- Test: `tests/test_frcnn_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frcnn_model.py
from train_faster_rcnn.model import build_fasterrcnn

def test_head_has_num_classes_plus_background():
    model = build_fasterrcnn(num_classes=4)  # 4 components
    # cls_score out_features == num_classes + 1 (background)
    assert model.roi_heads.box_predictor.cls_score.out_features == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frcnn_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `train_faster_rcnn/model.py`**

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int):
    """COCO-pretrained Faster R-CNN with the head resized to num_classes+1 (background)."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frcnn_model.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add train_faster_rcnn/model.py tests/test_frcnn_model.py
git commit -m "feat: Faster R-CNN model builder"
```

---

### Task 6: `train_faster_rcnn/train.py` — training CLI

**Files:**
- Create: `train_faster_rcnn/train.py`
- Test: `tests/test_frcnn_train.py`

- [ ] **Step 1: Write the failing test (one-step loop runs on CPU fixture)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frcnn_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `train_faster_rcnn/train.py`**

```python
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
    class_names = config.POLE_CLASSES if subset == "pole" else config.COMPONENT_CLASSES
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
    ap.add_argument("--subset", choices=["pole", "components"], required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.batch)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frcnn_train.py -v`
Expected: PASS (1 passed). (Downloads COCO weights on first run.)

- [ ] **Step 5: Commit**

```bash
git add train_faster_rcnn/train.py tests/test_frcnn_train.py
git commit -m "feat: Faster R-CNN training loop + CLI"
```

---

### Task 7: `train_rf_detr/layout.py` — rfdetr-compatible COCO view

**Files:**
- Create: `train_rf_detr/__init__.py`, `train_rf_detr/layout.py`
- Test: `tests/test_rfdetr_layout.py`

- [ ] **Step 1: Write the failing test**

RF-DETR expects `dataset/{train,valid,test}/_annotations.coco.json` with images alongside. We build that view from our COCO db via symlinks (no copy).

```python
# tests/test_rfdetr_layout.py
import json
from pathlib import Path
from train_rf_detr.layout import build_rfdetr_view

def test_builds_expected_structure(tmp_path):
    # fake our COCO db for one split
    src = tmp_path / "db" / "components"
    (src / "images" / "train" / "clahe").mkdir(parents=True)
    (src / "images" / "train" / "clahe" / "a.jpg").write_bytes(b"x")
    (src / "annotations").mkdir(parents=True)
    (src / "annotations" / "instances_train_clahe.json").write_text(
        json.dumps({"images": [{"id":1,"file_name":"a.jpg","width":2,"height":2}],
                    "annotations": [], "categories": [{"id":0,"name":"wire"}]}))
    out = build_rfdetr_view(src, version="clahe", dest=tmp_path / "rf")
    assert (Path(out) / "train" / "_annotations.coco.json").exists()
    assert (Path(out) / "train" / "a.jpg").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rfdetr_layout.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `train_rf_detr/layout.py`** (and empty `__init__.py`)

```python
import os
import shutil
from pathlib import Path

# rfdetr uses "valid" not "val"
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def build_rfdetr_view(subset_db: Path, version: str, dest: Path) -> Path:
    """Create dataset/{train,valid,test}/_annotations.coco.json + image symlinks
    from our COCO db, without copying image bytes."""
    subset_db, dest = Path(subset_db), Path(dest)
    for split, rf_split in _SPLIT_MAP.items():
        ann = subset_db / "annotations" / f"instances_{split}_{version}.json"
        if not ann.exists():
            continue
        out_dir = dest / rf_split
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ann, out_dir / "_annotations.coco.json")
        img_src = subset_db / "images" / split / version
        for img in img_src.glob("*.jpg"):
            link = out_dir / img.name
            if not link.exists():
                os.symlink(img.resolve(), link)
    return dest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rfdetr_layout.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add train_rf_detr/__init__.py train_rf_detr/layout.py tests/test_rfdetr_layout.py
git commit -m "feat: rfdetr-compatible COCO view via symlinks"
```

---

### Task 8: `train_rf_detr/train.py` — RF-DETR-L CLI

**Files:**
- Create: `train_rf_detr/train.py`
- Test: `tests/test_rfdetr_train.py`

- [ ] **Step 1: Write the failing test (param assembly + CUDA warning, no real train)**

```python
# tests/test_rfdetr_train.py
from unittest import mock
from train_rf_detr import train

def test_warns_when_not_cuda(capsys):
    with mock.patch("train_rf_detr.train.select_device", return_value="mps"), \
         mock.patch("train_rf_detr.train.build_rfdetr_view", return_value="/ds"), \
         mock.patch("train_rf_detr.train.RFDETRLarge") as M:
        train.run(subset="components", version="clahe", epochs=1, batch=2)
    assert "CUDA" in capsys.readouterr().out
    M.return_value.train.assert_called_once()
    kwargs = M.return_value.train.call_args.kwargs
    assert kwargs["dataset_dir"] == "/ds"
    assert kwargs["epochs"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rfdetr_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `train_rf_detr/train.py`**

```python
"""RF-DETR-L training (CUDA strongly preferred; use Colab). Usage:
    python -m train_rf_detr.train --subset components --version clahe --epochs 50 --batch 4
"""
import argparse
from pathlib import Path

from rfdetr import RFDETRLarge

from shared import config
from shared.device import select_device
from train_rf_detr.layout import build_rfdetr_view


def run(subset, version, epochs, batch):
    device = select_device()
    if device != "cuda":
        print(f"WARNING: device={device}. RF-DETR training is impractical off CUDA; "
              f"use the Colab notebook (Plan 3) for this model.")
    subset_db = config.COCO_DB / subset
    ds_dir = build_rfdetr_view(subset_db, version, Path(f"runs/{subset}/rfdetr_ds"))
    model = RFDETRLarge()
    return model.train(dataset_dir=str(ds_dir), epochs=epochs, batch_size=batch,
                       output_dir=f"runs/{subset}/rfdetr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pole", "components"], required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.batch)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rfdetr_train.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add train_rf_detr/train.py tests/test_rfdetr_train.py
git commit -m "feat: RF-DETR-L training CLI with CUDA-preference warning"
```

---

### Task 9: `inference/backends.py` — Detection + Detector + YOLO impl

**Files:**
- Create: `inference/__init__.py`, `inference/backends.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends.py
from inference.backends import Detection, parse_yolo_result, filter_detections

class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [xyxy]; self.conf = [conf]; self.cls = [cls]

class _FakeResult:
    def __init__(self, boxes, names): self.boxes = boxes; self.names = names

def test_parse_yolo_result_maps_names():
    res = _FakeResult([_FakeBox([0, 0, 10, 20], 0.9, 0)], {0: "pole"})
    dets = parse_yolo_result(res)
    assert dets == [Detection("pole", 0.9, (0, 0, 10, 20))]

def test_filter_by_confidence():
    dets = [Detection("wire", 0.2, (0, 0, 1, 1)), Detection("wire", 0.8, (0, 0, 1, 1))]
    assert filter_detections(dets, 0.5) == [Detection("wire", 0.8, (0, 0, 1, 1))]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `inference/backends.py`** (and empty `inference/__init__.py`)

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels of the image it was run on


def parse_yolo_result(result) -> list:
    """Convert one Ultralytics Result into Detection objects."""
    dets = []
    for b in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in list(b.xyxy[0]))
        cls = int(b.cls[0])
        dets.append(Detection(result.names[cls], float(b.conf[0]),
                              (x1, y1, x2, y2)))
    return dets


def filter_detections(dets, conf: float) -> list:
    return [d for d in dets if d.confidence >= conf]


class Detector(Protocol):
    def predict(self, image) -> list: ...


class YoloDetector:
    """Wraps an Ultralytics model behind the Detector interface."""

    def __init__(self, weights, conf=0.25, imgsz=1280):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, image) -> list:
        res = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        return parse_yolo_result(res)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add inference/__init__.py inference/backends.py tests/test_backends.py
git commit -m "feat: Detection type + Detector protocol + YOLO backend"
```

---

### Task 10: `inference/backends.py` — torchvision + RF-DETR backends

**Files:**
- Modify: `inference/backends.py`
- Test: `tests/test_backends_extra.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends_extra.py
import torch
from inference.backends import parse_torchvision_output, Detection

def test_parse_torchvision_output_1based_to_names():
    out = {"boxes": torch.tensor([[0., 0., 5., 5.]]),
           "scores": torch.tensor([0.7]),
           "labels": torch.tensor([1])}  # label 1 -> class_names[0]
    dets = parse_torchvision_output(out, class_names=["wire", "h_insulator"], conf=0.5)
    assert dets == [Detection("wire", 0.7, (0.0, 0.0, 5.0, 5.0))]

def test_parse_drops_below_conf():
    out = {"boxes": torch.tensor([[0., 0., 5., 5.]]),
           "scores": torch.tensor([0.2]), "labels": torch.tensor([1])}
    assert parse_torchvision_output(out, ["wire"], conf=0.5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends_extra.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Append to `inference/backends.py`**

```python
def parse_torchvision_output(output, class_names, conf: float) -> list:
    """Convert a torchvision detection dict into Detections (label 1 -> class_names[0])."""
    dets = []
    boxes = output["boxes"].tolist()
    scores = output["scores"].tolist()
    labels = output["labels"].tolist()
    for box, score, label in zip(boxes, scores, labels):
        if score < conf:
            continue
        idx = int(label) - 1  # undo background offset
        if 0 <= idx < len(class_names):
            dets.append(Detection(class_names[idx], float(score),
                                  tuple(float(v) for v in box)))
    return dets


class TorchvisionDetector:
    """Wraps a trained Faster R-CNN state_dict behind the Detector interface."""

    def __init__(self, weights_path, class_names, conf=0.5, device=None):
        import torch
        from shared.device import select_device
        from train_faster_rcnn.model import build_fasterrcnn
        self.class_names = class_names
        self.conf = conf
        self.device = device or select_device()
        self.model = build_fasterrcnn(len(class_names))
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval().to(self.device)

    def predict(self, image) -> list:
        import torch
        from torchvision.transforms.functional import to_tensor
        from PIL import Image
        import numpy as np
        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])  # BGR->RGB
        with torch.no_grad():
            out = self.model([to_tensor(image).to(self.device)])[0]
        out = {k: v.cpu() for k, v in out.items()}
        return parse_torchvision_output(out, self.class_names, self.conf)


class RFDetrDetector:
    """Wraps an RF-DETR-L checkpoint behind the Detector interface."""

    def __init__(self, weights_path, class_names, conf=0.5):
        from rfdetr import RFDETRLarge
        self.model = RFDETRLarge(pretrain_weights=weights_path)
        self.class_names = class_names
        self.conf = conf

    def predict(self, image) -> list:
        import supervision as sv  # rfdetr returns sv.Detections
        det = self.model.predict(image, threshold=self.conf)
        results = []
        for xyxy, conf, cls_id in zip(det.xyxy, det.confidence, det.class_id):
            results.append(Detection(self.class_names[int(cls_id)], float(conf),
                                     tuple(float(v) for v in xyxy)))
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends_extra.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add inference/backends.py tests/test_backends_extra.py
git commit -m "feat: torchvision + RF-DETR inference backends"
```

---

### Task 11: `inference/geometry.py` — crop & remap math

**Files:**
- Create: `inference/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `inference/geometry.py`**

```python
from inference.backends import Detection


def crop_with_pad(image, box, pad_frac=0.05):
    """Crop image to box (x1,y1,x2,y2) with optional padding fraction of box size.
    Returns (crop_array, (offset_x, offset_y)). Offsets are clamped to >= 0."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * pad_frac
    ph = (y2 - y1) * pad_frac
    cx1 = max(0, int(x1 - pw))
    cy1 = max(0, int(y1 - ph))
    cx2 = min(w, int(x2 + pw))
    cy2 = min(h, int(y2 + ph))
    return image[cy1:cy2, cx1:cx2], (cx1, cy1)


def shift_detection(det: Detection, off_x: int, off_y: int) -> Detection:
    """Map a detection from crop coordinates back to full-frame coordinates."""
    x1, y1, x2, y2 = det.box
    return Detection(det.class_name, det.confidence,
                     (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_geometry.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add inference/geometry.py tests/test_geometry.py
git commit -m "feat: crop-with-pad and crop->full detection remap"
```

---

### Task 12: `inference/pipeline.py` — two-stage chain → structured output

**Files:**
- Create: `inference/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test (fake detectors, no real models)**

```python
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
    # component is in CROP coords (relative to the 100x150 pole crop)
    comp_det = _Fake([Detection("wire", 0.8, (5, 5, 25, 25))])
    out = run_pipeline(img, pole_det, comp_det, crop_dir=tmp_path, image_name="x.jpg")
    assert len(out["poles"]) == 1
    pole = out["poles"][0]
    assert pole["confidence"] == 0.95
    comp = pole["components"][0]
    assert comp["class"] == "wire"
    # full-frame box = crop box + pole offset (10,10)
    assert comp["box_full"] == [15, 15, 35, 35]
    assert comp["crop_path"].endswith(".jpg")

def test_pipeline_handles_no_poles():
    img = np.zeros((50, 50, 3), np.uint8)
    out = run_pipeline(img, _Fake([]), _Fake([]), crop_dir=None, image_name="n.jpg")
    assert out["poles"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `inference/pipeline.py`**

```python
"""Two-stage inference: pole detect -> crop -> component detect -> crop each
component -> structured JSON. Usage:
    python -m inference.pipeline --image path.jpg --pole-weights p.pt --comp-weights c.pt
"""
import argparse
import json
from pathlib import Path

import cv2

from inference.backends import YoloDetector
from inference.geometry import crop_with_pad, shift_detection


def _save_crop(crop, crop_dir, name):
    if crop_dir is None or crop.size == 0:
        return None
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    path = crop_dir / name
    cv2.imwrite(str(path), crop)
    return str(path)


def run_pipeline(image, pole_detector, comp_detector, crop_dir, image_name):
    """image: BGR ndarray. Returns the structured result dict."""
    stem = Path(image_name).stem
    result = {"image": image_name, "poles": []}
    for pi, pole in enumerate(pole_detector.predict(image)):
        pole_crop, (ox, oy) = crop_with_pad(image, pole.box, pad_frac=0.05)
        pole_crop_path = _save_crop(pole_crop, crop_dir, f"{stem}_pole{pi}.jpg")
        components = []
        for ci, comp in enumerate(comp_detector.predict(pole_crop)):
            full = shift_detection(comp, ox, oy)
            comp_crop, _ = crop_with_pad(image, full.box, pad_frac=0.0)
            comp_crop_path = _save_crop(comp_crop, crop_dir, f"{stem}_pole{pi}_comp{ci}.jpg")
            components.append({
                "class": comp.class_name,
                "confidence": comp.confidence,
                "box_crop": [int(v) for v in comp.box],
                "box_full": [int(v) for v in full.box],
                "crop_path": comp_crop_path,
            })
        result["poles"].append({
            "box": [int(v) for v in pole.box],
            "confidence": pole.confidence,
            "crop_path": pole_crop_path,
            "components": components,
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pole-weights", required=True)
    ap.add_argument("--comp-weights", required=True)
    ap.add_argument("--crop-dir", default="runs/inference/crops")
    ap.add_argument("--out", default="runs/inference/result.json")
    a = ap.parse_args()
    image = cv2.imread(a.image)
    pole_det = YoloDetector(a.pole_weights)
    comp_det = YoloDetector(a.comp_weights)
    result = run_pipeline(image, pole_det, comp_det, a.crop_dir, Path(a.image).name)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add inference/pipeline.py tests/test_pipeline.py
git commit -m "feat: two-stage inference pipeline -> structured JSON"
```

---

### Task 13: `inference/infer_pole.py` & `infer_components.py` — single-model CLIs

**Files:**
- Create: `inference/infer_pole.py`, `inference/infer_components.py`
- Test: `tests/test_infer_single.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer_single.py
import numpy as np
from inference.backends import Detection
from inference.infer_pole import detections_to_records

def test_detections_to_records():
    dets = [Detection("pole", 0.9, (1, 2, 3, 4))]
    recs = detections_to_records(dets)
    assert recs == [{"class": "pole", "confidence": 0.9, "box": [1, 2, 3, 4]}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_infer_single.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `inference/infer_pole.py`**

```python
"""Run Model 1 (pole) on one image. Usage:
    python -m inference.infer_pole --image x.jpg --weights pole.pt
"""
import argparse
import json
from pathlib import Path

import cv2

from inference.backends import YoloDetector


def detections_to_records(dets):
    return [{"class": d.class_name, "confidence": d.confidence,
             "box": [int(v) for v in d.box]} for d in dets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    det = YoloDetector(a.weights, conf=a.conf)
    recs = detections_to_records(det.predict(cv2.imread(a.image)))
    print(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `inference/infer_components.py`** (imports the shared helper — DRY)

```python
"""Run Model 2 (components) on one image/crop. Usage:
    python -m inference.infer_components --image crop.jpg --weights components.pt
"""
import argparse
import json

import cv2

from inference.backends import YoloDetector
from inference.infer_pole import detections_to_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    det = YoloDetector(a.weights, conf=a.conf)
    recs = detections_to_records(det.predict(cv2.imread(a.image)))
    print(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_infer_single.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add inference/infer_pole.py inference/infer_components.py tests/test_infer_single.py
git commit -m "feat: single-model inference CLIs for pole and components"
```

---

### Task 14: Smoke-train + real pipeline validation (gate)

**Files:** none (validation gate; run on the M4 Pro / Colab)

- [ ] **Step 1: Full unit suite passes**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 2: 1-epoch YOLO smoke train (tiny) on the pole DB**

Run: `python -m train_yolo.train_pole --version clahe --epochs 1 --imgsz 640 --batch 2`
Expected: completes; writes `runs/pole/yolo/.../weights/best.pt`; prints whether YOLO26x or the yolo11x fallback was used.

- [ ] **Step 3: 1-epoch components smoke train**

Run: `python -m train_yolo.train_components --version clahe --epochs 1 --imgsz 640 --batch 2`
Expected: completes; writes `runs/components/yolo/.../weights/best.pt`.

- [ ] **Step 4: Run the full pipeline on one real image**

Run:
```bash
POLE=$(ls -t runs/pole/yolo*/weights/best.pt | head -1)
COMP=$(ls -t runs/components/yolo*/weights/best.pt | head -1)
python -m inference.pipeline \
  --image "/Volumes/dronisight/mem7/$(ls /Volumes/dronisight/mem7 | grep -m1 JPG)" \
  --pole-weights "$POLE" \
  --comp-weights "$COMP"
```
Expected: prints structured JSON with `poles[]`, each having `box`, `confidence`, `crop_path`, and `components[]` with `class`/`confidence`/`box_full`/`crop_path`; crop files written under `runs/inference/crops/`.

> Note: Ultralytics writes weights to `runs/{subset}/yolo/weights/best.pt` (and `runs/{subset}/yolo2/…` on re-runs, hence the `yolo*` glob + `ls -t`).

- [ ] **Step 5: Commit any fixups**

```bash
git add -A && git commit -m "test: smoke-train + end-to-end pipeline validated"
```

---

## Self-Review Notes (completed)
- **Spec coverage:** §1 three model families → Tasks 2–8; §6.1 crop-gap mitigation → Task 1 (`scale`) ; §6.3 aug policy → Task 1; §1 device priority → all CLIs via `select_device`; YOLO26x fallback (§5) → Task 2; inference per-model + pipeline + component crops (spec §5/§4) → Tasks 9–13.
- **Placeholders:** none — all code complete.
- **Type consistency:** `Detection` (Task 9) reused in Tasks 10–13; `select_device` (Plan 1 Task 2) used by every CLI; `build_fasterrcnn` (Task 5) reused in Tasks 6 and 10; `CocoDetectionDataset` (Task 4) used in Task 6; class-name lists flow from `config`.
- **Note:** `RFDetrDetector` (Task 10) imports `supervision` (an rfdetr transitive dep); if absent, add to deps in Task 0.
