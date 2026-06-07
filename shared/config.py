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
# Condition of a detected component (run on the above/below component crop). 14 classes;
# train is balanced to BALANCE_TARGET per class (cap the big ones, augment the small ones).
COMPONENT_CLASSIFICATION_CLASSES = [
    "straight_crossarm_normal", "straight_crossarm_band", "v_insulator_normal", "wire_normal",
    "h_insulator_normal", "cross_wire", "h_insulator_broken", "v_insulator_band",
    "v_insulator_broken", "top_crossarm_band", "h_insulator_chip_off", "om_crossarm_normal",
    "v_insulator_chip_off", "top_crossarm_normal",
]

# One place every consumer reads the per-subset class list from.
SUBSET_CLASSES = {
    "pole": POLE_CLASSES,
    "component_above_1000": COMPONENT_ABOVE_CLASSES,
    "component_below_1000": COMPONENT_BELOW_CLASSES,
    "component_classification": COMPONENT_CLASSIFICATION_CLASSES,
}
SUBSETS = list(SUBSET_CLASSES)

# Per-subset raw source folders: the mem captures feed pole/components; the '6th june'
# close-up condition captures feed component_classification.
_CONDITION_FOLDER_NAMES = [
    "6thMem1AllTeam1",
    "6thMem2AllTeam1/6thMem2AllTeam1",   # images nested one level below their XMLs
    "6thMem3AllTeam1", "6thMem4AllTeam1", "6thMem5AllTeam1",
    "6thMem6AllTeam1", "6thMem7AllTeam1", "6thMem8AllTeam1",
]
CONDITION_SOURCE_DIRS = [SSD_ROOT / "6th june " / n for n in _CONDITION_FOLDER_NAMES]
SUBSET_SOURCE_DIRS = {
    "pole": SOURCE_DIRS,
    "component_above_1000": SOURCE_DIRS,
    "component_below_1000": SOURCE_DIRS,
    "component_classification": CONDITION_SOURCE_DIRS,
}

# Per-subset TRAIN class-balance target (instances/class): cap classes above it, augment below.
# Subsets not listed use cap-to-rarest (pole/above) or oversample-to-max (below).
BALANCE_TARGET = {"component_classification": 400}

# Split
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

# Grouping: new capture-sequence group when consecutive frames differ by > this (seconds)
GROUP_TIME_GAP_S = 60

# Balancing
BALANCE_CAP_ENABLED = True  # cap to lowest kept-class count per sub-dataset

# CLAHE defaults (per-image params still come from the profile)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
