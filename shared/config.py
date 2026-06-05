import os
from pathlib import Path

SEED = 1337

# Root holding the source data and/or the built DBs.
# Override with the DRONISIGHT_DATA env var (e.g. when training on the M4 from a
# local copy of the DBs); defaults to the external SSD mount used during data-prep.
SSD_ROOT = Path(os.environ.get("DRONISIGHT_DATA", "/Volumes/dronisight"))
# Raw source folders (names contain spaces). Edit this list to add/remove captures.
_SOURCE_FOLDER_NAMES = [
    "mem2 5th june", "mem3", "mem4 5th june", "mem5", "mem6", "mem7", "mem8",
    "mem 7.1 5th june", "mem10", "4thJuneMem4", "4thJuneMem8",
]
SOURCE_DIRS = [SSD_ROOT / name for name in _SOURCE_FOLDER_NAMES]

# Annotation dedup: when the same image (matched by filename stem) appears in both
# folders of a pair with an IDENTICAL annotation hash, drop the copy in the second
# folder. Different annotations for the same stem are kept (they get distinct keys).
DEDUP_PAIRS = [("mem7", "mem 7.1 5th june")]

# Output DBs
YOLO_DB = SSD_ROOT / "yolo_train_db"
COCO_DB = SSD_ROOT / "RF_DETR_Faster_RCNN_train_db"

# Class policy -> three detectors:
#   pole                  : the pole on the full frame
#   component_above_1000  : the 4 high-frequency component classes (>1000 instances)
#   component_below_1000  : the 4 rare component classes (<1000) -- now trained (oversampled),
#                           previously ignored. Still never deleted from source.
POLE_CLASSES = ["pole"]
COMPONENT_ABOVE_CLASSES = ["wire", "h_insulator", "v_insulator", "crossarm_stright"]
COMPONENT_BELOW_CLASSES = ["vegetation", "top_crossarm", "om_crossarm", "rust"]

# One place every consumer reads the per-subset class list from.
SUBSET_CLASSES = {
    "pole": POLE_CLASSES,
    "component_above_1000": COMPONENT_ABOVE_CLASSES,
    "component_below_1000": COMPONENT_BELOW_CLASSES,
}
SUBSETS = list(SUBSET_CLASSES)

# Split
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

# Grouping: new capture-sequence group when consecutive frames differ by > this (seconds)
GROUP_TIME_GAP_S = 60

# Balancing
BALANCE_CAP_ENABLED = True  # cap to lowest kept-class count per sub-dataset

# CLAHE defaults (per-image params still come from the profile)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
