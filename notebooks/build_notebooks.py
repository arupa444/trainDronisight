"""Generate Colab notebooks from cell specs. Run: python -m notebooks.build_notebooks"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

REPO_URL = "https://github.com/arupa444/trainDronisight.git"  # set before publishing

_SETUP = [
    "# @title Setup: GPU check + repo + deps",
    "!nvidia-smi",
    f"from pathlib import Path\n"
    f"REPO='/content/repo'\n"
    f"!git clone {REPO_URL} $REPO 2>/dev/null || (cd $REPO && git pull)\n"
    "%cd $REPO\n"
    "!pip -q install uv && uv pip install --system -e .",
    "import torch; print('CUDA:', torch.cuda.is_available())",
    "from notebooks.colab_utils import mount_drive, drive_db_zip, ensure_dataset\nmount_drive()",
    # shared.config defaults to /Volumes/dronisight (the M4's SSD mount). On Colab the
    # DBs are unzipped to /content/data, so point DRONISIGHT_DATA there or every module
    # that reads config.YOLO_DB / config.COCO_DB (verify + all trainers) hits a missing path.
    "import os; os.environ['DRONISIGHT_DATA'] = '/content/data'  # matches the ensure_dataset() dest below",
]

NOTEBOOKS = {
    "00_data_prep": _SETUP + [
        "# Unzip both DBs from Drive to fast local storage\n"
        "ensure_dataset(drive_db_zip('yolo_train_db'), '/content/data', 'yolo_train_db')\n"
        "ensure_dataset(drive_db_zip('RF_DETR_Faster_RCNN_train_db'), '/content/data', 'RF_DETR_Faster_RCNN_train_db')",
        "# (Optional) Re-run data-prep from raw mem*/6th-june if they're on Drive instead of prebuilt DBs.\n"
        "# build_dataset content-hash-merges byte-identical photo copies (the per-annotator 6th-june\n"
        "# data has the same photo in several member folders) -> one entry with the UNION of all boxes.\n"
        "# !python -m data_prep.build_dataset --subset all",
        "# verify asserts no capture-group AND no image-content leakage across splits\n"
        "!python -m data_prep.verify_dataset --subset pole\n"
        "!python -m data_prep.verify_dataset --subset component_above_1000\n"
        "!python -m data_prep.verify_dataset --subset component_below_1000\n"
        "!python -m data_prep.verify_dataset --subset component_classification",
    ],
    "01_train_yolo": _SETUP + [
        "ensure_dataset(drive_db_zip('yolo_train_db'), '/content/data', 'yolo_train_db')",
        "# 1) pole: fills the frame -> imgsz 640 (matches train_pole.py default)\n"
        "!python -m train_yolo.train_pole --version clahe --epochs 100 --imgsz 640 --batch 16",
        "# 2) component_above_1000 (wire/h_insulator/v_insulator/crossarm_stright): 1280 for thin wires\n"
        "!python -m train_yolo.train_components --subset component_above_1000 --version clahe --epochs 150 --imgsz 1280 --batch 16 --model yolo26m.pt",
        "# 3) component_below_1000 (vegetation/top_crossarm/om_crossarm/rust): oversampled train, more epochs\n"
        "!python -m train_yolo.train_components --subset component_below_1000 --version clahe --epochs 200 --imgsz 1280 --batch 16 --model yolo26m.pt",
        "# 4) component_classification (14 condition classes; train balanced to ~400/class)\n"
        "!python -m train_yolo.train_components --subset component_classification --version clahe --epochs 150 --imgsz 1280 --batch 16 --model yolo26m.pt",
        "# 5) (ablation) crop-aligned variants: trained on pole/component CROPS so train scale ==\n"
        "#    inference scale. Needs the *_crop datasets (build_dataset --subset all_crop). Compare\n"
        "#    each crop model's val mAP to its full-frame twin and run the winner.\n"
        "!python -m train_yolo.train_components --subset component_above_1000_crop --version clahe --epochs 150 --imgsz 1280 --batch 16 --model yolo26m.pt\n"
        "!python -m train_yolo.train_components --subset component_below_1000_crop --version clahe --epochs 200 --imgsz 1280 --batch 16 --model yolo26m.pt\n"
        "!python -m train_yolo.train_components --subset component_classification_crop --version clahe --epochs 150 --imgsz 1280 --batch 16 --model yolo26m.pt",
        "# Colab runtimes are ephemeral -> copy weights + plots to Drive before the session ends\n"
        "from notebooks.colab_utils import save_runs_to_drive\nprint('saved to', save_runs_to_drive())",
    ],
    "02_train_faster_rcnn": _SETUP + [
        "ensure_dataset(drive_db_zip('RF_DETR_Faster_RCNN_train_db'), '/content/data', 'RF_DETR_Faster_RCNN_train_db')",
        "!python -m train_faster_rcnn.train --subset pole --version clahe --epochs 30 --batch 4",
        "!python -m train_faster_rcnn.train --subset component_above_1000 --version clahe --epochs 30 --batch 4",
        "!python -m train_faster_rcnn.train --subset component_below_1000 --version clahe --epochs 30 --batch 4",
        "!python -m train_faster_rcnn.train --subset component_classification --version clahe --epochs 60 --batch 2",
        "from notebooks.colab_utils import save_runs_to_drive\nprint('saved to', save_runs_to_drive())",
    ],
    "03_train_rf_detr": _SETUP + [
        "ensure_dataset(drive_db_zip('RF_DETR_Faster_RCNN_train_db'), '/content/data', 'RF_DETR_Faster_RCNN_train_db')",
        "# --version clahe trains on CLAHE pixels; --resolution must be a multiple of the model\n"
        "# block_size (patch_size*num_windows: 32 on the current RF-DETR build). 672/896/1120 are\n"
        "# multiples of BOTH 32 and 56, so they're safe across lib versions and need no predict rounding.\n"
        "!python -m train_rf_detr.train --subset pole --version clahe --epochs 50 --batch 4 --resolution 672",
        "!python -m train_rf_detr.train --subset component_above_1000 --version clahe --epochs 50 --batch 4 --resolution 672",
        "!python -m train_rf_detr.train --subset component_below_1000 --version clahe --epochs 50 --batch 4 --resolution 672",
        "# higher res for the 14-class condition stage (more small-object detail)\n"
        "!python -m train_rf_detr.train --subset component_classification --version clahe --epochs 50 --batch 8 --resolution 1120",
        "from notebooks.colab_utils import save_runs_to_drive\nprint('saved to', save_runs_to_drive())",
    ],
    "04_inference_pipeline": _SETUP + [
        "ensure_dataset(drive_db_zip('yolo_train_db'), '/content/data', 'yolo_train_db')",
        "# fresh runtime? pull previously-trained weights back from Drive into runs/\n"
        "from notebooks.colab_utils import restore_runs_from_drive\nprint('restored', restore_runs_from_drive(), 'files from Drive')",
        "# the 12 weights (pole + 5 component + 6 condition specialists) are auto-discovered by\n"
        "# subset name under runs/ (restored above) — no per-model flags needed\n"
        "from shared import config\nfrom inference.pipeline import discover_weights\n"
        "found=discover_weights('runs', config.SUBSETS)\nprint({s: s in found for s in config.SUBSETS})",
        "import glob\nIMG=sorted(glob.glob('/content/data/yolo_train_db/comp_insulator/images/test/clahe/*.jpg'))[0]\nprint(IMG)",
        "# full 12-model routing: pole -> 5 component specialists -> NMS -> route each to its\n"
        "# condition specialist. result.csv + result.json + viz/{pole,components,conditions,all}/\n"
        "!python -m inference.pipeline --image \"$IMG\" --weights-dir runs --out-dir /content/inference",
        "import glob, os, json\nrun=sorted(glob.glob('/content/inference/*_inference*'), key=os.path.getmtime)[-1]\n"
        "print('run dir:', run)\nprint(json.dumps(json.load(open(f'{run}/result.json')), indent=2)[:2000])",
        "# preview the annotated 'all' views\n"
        "from IPython.display import Image, display\n"
        "[display(Image(filename=p, width=900)) for p in sorted(glob.glob(f'{run}/viz/all/*.jpg'))[:5]]",
        "# persist the whole run (csv/json/crops/viz) to Drive\n"
        "import shutil\nshutil.copytree(run, f\"/content/drive/MyDrive/finalGo/inference/{os.path.basename(run)}\", dirs_exist_ok=True)\nprint('saved to Drive')",
    ],
}


def _to_notebook(title, cells):
    # Deterministic, content-stable cell ids so regenerating (e.g. after the user
    # sets REPO_URL) yields a clean diff instead of churning random ids each run.
    nb = new_notebook()
    md = new_markdown_cell(f"# {title}\n\nGenerated by `notebooks/build_notebooks.py`. "
                           "Runtime → Change runtime type → GPU, then Run all.")
    md["id"] = f"{title}-md"
    nb.cells.append(md)
    for i, c in enumerate(cells):
        cell = new_code_cell(c)
        cell["id"] = f"{title}-cell-{i}"
        nb.cells.append(cell)
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": []}
    return nb


def build_all(out_dir=None):
    out_dir = Path(out_dir or Path(__file__).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = _to_notebook(name, cells)
        path = out_dir / f"{name}.ipynb"
        nbformat.write(nb, str(path))
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    for p in build_all():
        print("wrote", p)
