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
