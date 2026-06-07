"""RF-DETR-L training (CUDA strongly preferred; use Colab). Usage:
    python -m train_rf_detr.train --subset component_above_1000 --version clahe \
        --epochs 50 --batch 4 --resolution 728

CLAHE: --version clahe makes build_rfdetr_view symlink the `clahe` image variant, so the
model trains on the same CLAHE-preprocessed pixels as YOLO/FRCNN. --resolution (multiple
of 56) sets RF-DETR's input size; the library default (~560) is too low for thin wires,
so raise it (728 / 1008) for a fair small-object comparison vs YOLO@1280.
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


def run(subset, version, epochs, batch, resolution=728):
    if resolution % 56 != 0:
        raise ValueError(f"RF-DETR resolution must be divisible by 56 (got {resolution}); "
                         f"try {round(resolution / 56) * 56}.")
    device = select_device()
    if device != "cuda":
        print(f"WARNING: device={device}. RF-DETR training is impractical off CUDA; "
              f"use the Colab notebook (Plan 3) for this model.")
    subset_db = config.COCO_DB / subset
    # build_rfdetr_view symlinks the chosen image VARIANT -> 'clahe' = CLAHE-preprocessed pixels
    ds_dir = build_rfdetr_view(subset_db, version, Path(f"runs/{subset}/rfdetr_ds"))

    print("=" * 64)
    print(f"[rfdetr] subset={subset}  version={version}  device={device}")
    print(f"[rfdetr] classes: {config.SUBSET_CLASSES.get(subset, '?')}")
    print(f"[rfdetr] CLAHE preprocessing: {'ON (clahe variant)' if version == 'clahe' else 'OFF (orig variant)'}")
    print(f"[rfdetr] resolution={resolution} (%56==0)  epochs={epochs}  batch={batch}  "
          f"(lib default ~560 raised for thin wires)")
    print(f"[rfdetr] dataset_view={ds_dir}  ->  outputs runs/{subset}/rfdetr")
    print("=" * 64, flush=True)

    model = RFDETRLarge(resolution=resolution)
    return model.train(dataset_dir=str(ds_dir), epochs=epochs, batch_size=batch,
                       output_dir=f"runs/{subset}/rfdetr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=config.SUBSETS, required=True)
    ap.add_argument("--version", choices=["orig", "clahe"], default="clahe")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=728,
                    help="input size, must be divisible by 56 (default 728; 1008 for more "
                         "small-object detail if the GPU has the memory)")
    a = ap.parse_args()
    run(a.subset, a.version, a.epochs, a.batch, a.resolution)


if __name__ == "__main__":
    main()
