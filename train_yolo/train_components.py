"""Train a component detector. Usage:
    python -m train_yolo.train_components --subset component_above_1000 --version clahe \
        --epochs 150 --imgsz 1280 --batch 16 --model yolo26m.pt
    python -m train_yolo.train_components --subset component_below_1000 --version clahe \
        --epochs 200 --imgsz 1280 --batch 16 --model yolo26m.pt

Two component subsets share this trainer:
  * component_above_1000 : wire, h_insulator, v_insulator, crossarm_stright (balance-capped)
  * component_below_1000 : vegetation, top_crossarm, om_crossarm, rust (offline-oversampled)
Components include thin wires, so keep imgsz high (1280). yolo26x at 1280 needs a lot of
MPS memory; use --model yolo26m.pt / yolo26l.pt and/or lower --batch if you hit OOM.
"""
import argparse
from ultralytics import YOLO
from shared import config
from shared.device import select_device
from shared.train_args import build_yolo_args
from train_yolo.weights import resolve_weights
from data_prep.emit_yolo import write_data_yaml

# the unified component detector + the 6 per-family condition specialists (all non-pole subsets)
COMPONENT_SUBSETS = ["component", "cond_v_insulator", "cond_h_insulator", "cond_straight_crossarm",
                     "cond_top_crossarm", "cond_om_crossarm", "cond_wire"]


def run(subset, version, epochs, imgsz, batch, model="yolo26x.pt"):
    weights, fell_back = resolve_weights(model, "yolo11x.pt")
    if fell_back:
        print("WARNING: using yolo11x fallback weights.")
    device = select_device()
    class_names = config.SUBSET_CLASSES[subset]
    data_dir = config.YOLO_DB / subset
    # Regenerate data.yaml so its `path:` points at the CURRENT DB location (the
    # build-time absolute path would otherwise send YOLO back to the build machine).
    data_yaml = (write_data_yaml(data_dir, version, class_names)
                 if data_dir.is_dir() else str(data_dir / f"data_{version}.yaml"))
    args = build_yolo_args(subset, data_yaml, device, epochs, imgsz, batch)
    yolo = YOLO(weights)
    return yolo.train(**args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=COMPONENT_SUBSETS, required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--model", default="yolo26x.pt",
                    help="preferred weights, e.g. yolo26m.pt / yolo26l.pt / yolo26x.pt")
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.imgsz, a.batch, a.model)


if __name__ == "__main__":
    main()
