"""Evaluate a trained Faster R-CNN with COCO mAP + per-class AP. Usage:
    python -m train_faster_rcnn.eval --subset component_above_1000 \
        --weights runs/component_above_1000/faster_rcnn/best.pt --split test

val_loss tells you WHEN the model generalizes best; this tells you HOW GOOD it is
(and lets you compare fairly against YOLO). Reports AP@[.5:.95], AP@.5 overall + per class.
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


def to_coco_dets(image_id, boxes, scores, labels, conf):
    """Convert one image's model output to COCO-results dicts: 1-based label -> 0-based
    category_id (matching our GT json), xyxy -> xywh, drop below `conf`."""
    out = []
    for box, score, label in zip(boxes, scores, labels):
        if float(score) < conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        out.append({"image_id": int(image_id), "category_id": int(label) - 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return out


@torch.no_grad()
def collect_detections(model, loader, device, conf=0.05):
    model.eval().to(device)
    dets = []
    for images, targets in loader:
        ids = [int(t["image_id"].item()) for t in targets]
        outs = model([im.to(device) for im in images])
        for img_id, out in zip(ids, outs):
            dets += to_coco_dets(img_id, out["boxes"].cpu(), out["scores"].cpu(),
                                 out["labels"].cpu(), conf)
    return dets


def _per_class_ap(coco_eval, coco_gt):
    """AP@[.5:.95] and AP@.5 per category from COCOeval.eval['precision'] [T,R,K,A,M]."""
    prec = coco_eval.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    rows = []
    for k, cid in enumerate(cat_ids):
        p = prec[:, :, k, 0, -1]          # all IoU, all recall, area=all, maxDet=100
        p50 = prec[0, :, k, 0, -1]        # IoU=0.5
        ap = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
        ap50 = float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")
        rows.append((names[cid], ap, ap50))
    return rows


def evaluate(subset, version, weights, split="test", conf=0.05):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    device = select_device()
    class_names = config.SUBSET_CLASSES[subset]
    img_dir = config.COCO_DB / subset / "images" / split / version
    ann = config.COCO_DB / subset / "annotations" / f"instances_{split}_{version}.json"
    ds = CocoDetectionDataset(img_dir, ann)            # augment=False (clean eval)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate)

    model = build_fasterrcnn(num_classes=len(class_names))
    model.load_state_dict(torch.load(weights, map_location=device))
    dets = collect_detections(model, dl, device, conf=conf)
    print(f"[eval] {subset} {split}: {len(ds)} images, {len(dets)} detections "
          f"(conf>={conf}) on device={device}")
    if not dets:
        print("[eval] no detections above threshold — model is likely undertrained.")
        return None

    coco_gt = COCO(str(ann))
    coco_eval = COCOeval(coco_gt, coco_gt.loadRes(dets), "bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    print(f"\n[eval] per-class AP ({subset}, {split}):")
    print(f"  {'class':18s}{'AP@.5:.95':>11s}{'AP@.5':>9s}")
    for name, ap, ap50 in _per_class_ap(coco_eval, coco_gt):
        print(f"  {name:18s}{ap:>11.3f}{ap50:>9.3f}")
    return coco_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=config.SUBSETS, required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--weights", default=None,
                    help="defaults to runs/<subset>/faster_rcnn/best.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    weights = a.weights or f"runs/{a.subset}/faster_rcnn/best.pt"
    evaluate(a.subset, a.version, weights, a.split, a.conf)


if __name__ == "__main__":
    main()
