"""RF-DETR-L training (CUDA strongly preferred; use Colab). Usage:
    python -m train_rf_detr.train --subset components --version clahe --epochs 50 --batch 4
"""
import argparse
from pathlib import Path

try:
    from rfdetr import RFDETRLarge
except ImportError:
    RFDETRLarge = None  # installed on Colab/CUDA; mocked in tests here

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
    ap.add_argument("--subset", choices=config.SUBSETS, required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.batch)


if __name__ == "__main__":
    main()
