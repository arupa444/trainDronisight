"""Per-class validation of a trained YOLO model against a subset's labeled val/test split.

Usage (run with the SSD mounted so the weights + DB are reachable):
    python -m train_yolo.eval_yolo --weights runs/pole/yolo/weights/best.pt \
        --subset pole --split val --imgsz 640

Prints overall mAP50 / mAP50-95 / P / R and per-class AP50/P/R, and APPENDS the same report to
--out (default runs/logs/eval_all.txt) so several models accumulate in one file. The data.yaml is
regenerated to the current DB path, so it works wherever the DB lives.
"""
import argparse
from pathlib import Path

from shared import config
from shared.device import select_device
from data_prep.emit_yolo import write_data_yaml


def evaluate(weights, subset, version="clahe", split="val", imgsz=1280, device=None, batch=4):
    from ultralytics import YOLO
    data_dir = config.YOLO_DB / subset
    data_yaml = write_data_yaml(data_dir, version, config.SUBSET_CLASSES[subset])
    res = YOLO(weights).val(data=data_yaml, split=split, imgsz=imgsz,
                            device=device or select_device(), batch=batch,
                            verbose=False, plots=False)
    return format_report(res, weights, subset, split)


def format_report(res, weights, subset, split):
    b = res.box
    lines = [f"### {subset} [{split}]  weights={weights}",
             f"    overall  mAP50={b.map50:.3f}  mAP50-95={b.map:.3f}  P={b.mp:.3f}  R={b.mr:.3f}"]
    for i, ci in enumerate(b.ap_class_index):
        lines.append(f"    {res.names[ci]:26s} AP50={b.ap50[i]:.3f}  P={b.p[i]:.3f}  R={b.r[i]:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--subset", required=True, choices=config.SUBSETS)
    ap.add_argument("--version", default="clahe", choices=["orig", "clahe"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=1280, help="MUST match training (pole 640, components 1280)")
    ap.add_argument("--device", default=None, help="e.g. mps / cuda / cpu (default: auto)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="runs/logs/eval_all.txt")
    a = ap.parse_args()
    report = evaluate(a.weights, a.subset, a.version, a.split, a.imgsz, a.device, a.batch)
    print(report)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(report + "\n\n")
    print(f"\n(appended to {a.out})")


if __name__ == "__main__":
    main()
