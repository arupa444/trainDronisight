import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter
from torch.utils.data import Dataset

# Train-time photometric jitter (boxes unaffected). Geometric aug = horizontal flip only
# (no vertical flip — poles/components have a strong up-down orientation prior).
_JITTER = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03)


class CocoDetectionDataset(Dataset):
    """Reads our COCO db. Converts [x,y,w,h] cat_id(0-based) ->
    xyxy boxes + 1-based labels (torchvision reserves 0 for background).
    With augment=True (train only), applies bbox-aware horizontal flip + color jitter."""

    def __init__(self, images_dir, ann_json, augment=False):
        self.images_dir = Path(images_dir)
        self.augment = augment
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
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.augment:
            img = _JITTER(img)                       # photometric: boxes unchanged
            if random.random() < 0.5:                # horizontal flip: remap x-coords
                W = img.width
                img = TF.hflip(img)
                if boxes.numel():
                    x1 = boxes[:, 0].clone()
                    boxes[:, 0] = W - boxes[:, 2]
                    boxes[:, 2] = W - x1

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return TF.to_tensor(img), target
