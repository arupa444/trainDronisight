import os
from pathlib import Path

SEED = 1337

# Root holding the source data and/or the built DBs.
# Override with the DRONISIGHT_DATA env var (e.g. when training on the M4 from a
# local copy of the DBs); defaults to the external SSD mount used during data-prep.
SSD_ROOT = Path(os.environ.get("DRONISIGHT_DATA", "/Volumes/dronisight"))
SOURCE_DIRS = [SSD_ROOT / f"mem{i}" for i in range(2, 9)]  # mem1 has no labels

# Output DBs
YOLO_DB = SSD_ROOT / "yolo_train_db"
COCO_DB = SSD_ROOT / "RF_DETR_Faster_RCNN_train_db"

# Class policy (>1000-instance classes only; others ignored, never deleted)
POLE_CLASSES = ["pole"]
COMPONENT_CLASSES = ["wire", "h_insulator", "v_insulator", "crossarm_stright"]

# Split
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

# Grouping: new capture-sequence group when consecutive frames differ by > this (seconds)
GROUP_TIME_GAP_S = 60

# Balancing
BALANCE_CAP_ENABLED = True  # cap to lowest kept-class count per sub-dataset

# CLAHE defaults (per-image params still come from the profile)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
