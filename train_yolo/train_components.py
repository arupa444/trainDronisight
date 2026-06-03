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
