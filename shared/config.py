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

# Class policy -> pole + ONE unified component detector + per-family condition specialists:
#   pole       : the pole on the full frame.
#   component  : ALL component TYPES in ONE detector, run on the pole crop. Unified (not split
#                above/below by frequency) so the three crossarm types compete in one softmax and
#                stop being mis-typed (om_crossarm <-> straight crossarm), and so the model learns
#                negatives across all types.
#   cond_*     : per-component-family CONDITION specialists, each run on a crop of one detected
#                component of that family (the old single 14-class condition model is split so each
#                specialist masters its family's subtle defects; insulators were the weak spot).
POLE_CLASSES = ["pole"]
COMPONENT_CLASSES = ["wire", "h_insulator", "v_insulator", "crossarm_stright",
                     "top_crossarm", "om_crossarm", "vegetation", "rust"]

# Condition specialists, one per base component family (om_crossarm_band added per the data).
COND_V_INSULATOR_CLASSES       = ["v_insulator_normal", "v_insulator_band", "v_insulator_broken", "v_insulator_chip_off"]
COND_H_INSULATOR_CLASSES       = ["h_insulator_normal", "h_insulator_broken", "h_insulator_chip_off"]
COND_STRAIGHT_CROSSARM_CLASSES = ["straight_crossarm_normal", "straight_crossarm_band"]
COND_TOP_CROSSARM_CLASSES      = ["top_crossarm_normal", "top_crossarm_band"]
COND_OM_CROSSARM_CLASSES       = ["om_crossarm_normal", "om_crossarm_band"]
COND_WIRE_CLASSES              = ["wire_normal", "cross_wire"]

# Which condition specialist each detected component routes to at inference (None = no condition).
# `crossarm_stright` (detector name) -> the straight_crossarm condition family.
COMPONENT_TO_CONDITION_MODEL = {
    "v_insulator":      "cond_v_insulator",
    "h_insulator":      "cond_h_insulator",
    "crossarm_stright": "cond_straight_crossarm",
    "top_crossarm":     "cond_top_crossarm",
    "om_crossarm":      "cond_om_crossarm",
    "wire":             "cond_wire",
    # vegetation, rust -> no condition family (presence/defect only)
}

# One place every consumer reads the per-subset class list from.
SUBSET_CLASSES = {
    "pole": POLE_CLASSES,
    "component": COMPONENT_CLASSES,
    "cond_v_insulator": COND_V_INSULATOR_CLASSES,
    "cond_h_insulator": COND_H_INSULATOR_CLASSES,
    "cond_straight_crossarm": COND_STRAIGHT_CROSSARM_CLASSES,
    "cond_top_crossarm": COND_TOP_CROSSARM_CLASSES,
    "cond_om_crossarm": COND_OM_CROSSARM_CLASSES,
    "cond_wire": COND_WIRE_CLASSES,
}
SUBSETS = list(SUBSET_CLASSES)
COND_SUBSETS = ["cond_v_insulator", "cond_h_insulator", "cond_straight_crossarm",
                "cond_top_crossarm", "cond_om_crossarm", "cond_wire"]

# component -> the condition classes valid for it (derived from the per-family models); used by the
# pipeline to keep only in-family condition detections. vegetation/rust absent -> no condition.
COMPONENT_TO_CONDITIONS = {c: SUBSET_CLASSES[m] for c, m in COMPONENT_TO_CONDITION_MODEL.items()}

# Crop padding (single source of truth so BUILD scale == inference scale):
POLE_CROP_PAD = 0.05        # `component` detector: crop to the pole box + this pad (== inference --pole-pad)
CONDITION_CROP_PAD = 0.25   # `cond_*` specialists: crop to the component box + this pad (== inference --condition-pad)

# How each subset is crop-aligned at BUILD time. A subset is crop-trained IFF it appears here.
#   "anchor": crop to each anchor (pole) box + pad; keep in-subset boxes >= min_visible.
#   "self":   crop to each in-subset box + pad (the component itself).
# `pole` is absent -> trained on the full frame.
CROP_ALIGN = {
    "component": ("anchor", tuple(POLE_CLASSES), POLE_CROP_PAD, 0.30),
    **{s: ("self", None, CONDITION_CROP_PAD, 0.50) for s in COND_SUBSETS},
}

# Per-subset raw source folders: the mem captures feed pole + component; the '6th june'
# close-up condition captures feed every cond_* specialist.
_CONDITION_FOLDER_NAMES = [
    "6thMem1AllTeam1",
    "6thMem2AllTeam1/6thMem2AllTeam1",   # images nested one level below their XMLs
    "6thMem3AllTeam1", "6thMem4AllTeam1", "6thMem5AllTeam1",
    "6thMem6AllTeam1", "6thMem7AllTeam1", "6thMem8AllTeam1",
]
CONDITION_SOURCE_DIRS = [SSD_ROOT / "6th june " / n for n in _CONDITION_FOLDER_NAMES]
SUBSET_SOURCE_DIRS = {
    "pole": SOURCE_DIRS,
    "component": SOURCE_DIRS,
    **{s: CONDITION_SOURCE_DIRS for s in COND_SUBSETS},
}

# Per-subset TRAIN class-balance target (instances/class): cap classes above it, augment below it;
# val/test stay raw. `component` caps the frequent classes (wire ~3.4k) down and augments the rare
# ones (rust ~225) up; each condition specialist balances its 2-4 classes to 400.
BALANCE_TARGET = {
    "component": 1500,
    **{s: 400 for s in COND_SUBSETS},
}

# Collapse every byte-identical copy of one physical image into a SINGLE entry holding
# the UNION of all copies' boxes, BEFORE splitting. This is keyed on image CONTENT hash,
# so it is safe everywhere: it only ever merges genuinely identical photos and can never
# fuse two distinct captures. It is REQUIRED for the 6th-june condition data (8-9 members
# each labeled different classes over the same ~9k photos -> one photo lives in several
# member folders with only partial labels), and it also fixes a latent form of the same
# bug in the mem7/mem7.1 overlap (byte-identical photos that carried different annotations
# were previously kept as two partial-label, split-leaking copies). Enabled for every
# subset; it is a no-op on truly disjoint captures (just a hashing pass).
MERGE_CROSS_FOLDER = {s: True for s in SUBSETS}

# For the CONDITION specialists, after merging the union of members' boxes, resolve the case where
# members gave the SAME physical object DIFFERENT condition labels: defect beats normal, and a
# defect-vs-defect disagreement is dropped as ambiguous. ON for every cond_* (for the `component`
# detector, overlapping distinct types like a wire crossing a crossarm are legitimate -> OFF).
RESOLVE_CONDITION_CONFLICTS = {s: True for s in COND_SUBSETS}

# Split
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

# Grouping: new capture-sequence group when consecutive frames differ by > this (seconds)
GROUP_TIME_GAP_S = 60

# Balancing
BALANCE_CAP_ENABLED = True  # cap to lowest kept-class count per sub-dataset

# CLAHE defaults (per-image params still come from the profile)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
