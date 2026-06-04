"""Train Model 1 (pole). Usage:
    python -m train_yolo.train_pole --version clahe --epochs 100 --imgsz 640 --batch 4
Poles fill most of the frame, so a smaller imgsz (640-960) is plenty and avoids MPS OOM.
Pick a lighter model with --model yolo26m.pt / yolo26l.pt if memory is tight.
"""
import argparse
from ultralytics import YOLO
from shared import config
from shared.device import select_device
from shared.train_args import build_yolo_args
from train_yolo.weights import resolve_weights


def run(version, epochs, imgsz, batch, model="yolo26x.pt"):
    weights, fell_back = resolve_weights(model, "yolo11x.pt")
    if fell_back:
        print("WARNING: using yolo11x fallback weights.")
    device = select_device()
    data_yaml = str(config.YOLO_DB / "pole" / f"data_{version}.yaml")
    args = build_yolo_args("pole", data_yaml, device, epochs, imgsz, batch)
    yolo = YOLO(weights)
    return yolo.train(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--model", default="yolo26x.pt",
                    help="preferred weights, e.g. yolo26m.pt / yolo26l.pt / yolo26x.pt")
    a = ap.parse_args()
    run(a.version, a.epochs, a.imgsz, a.batch, a.model)


if __name__ == "__main__":
    main()
