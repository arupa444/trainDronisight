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

# Component -> the condition classes that are VALID for it. At inference the condition model
# runs on a CROP of one detected component, so a v_insulator crop can only carry v_insulator_*
# conditions (never a crossarm/wire condition). The pipeline maps each component to its family
# and DROPS condition detections outside it. Components with no condition family (vegetation,
# rust) get no condition. Note: the detector class `crossarm_stright` maps to the `straight_crossarm_*`
# condition names. These 6 families partition all 14 COMPONENT_CLASSIFICATION_CLASSES exactly.
COMPONENT_TO_CONDITIONS = {
    "v_insulator":      ["v_insulator_normal", "v_insulator_band", "v_insulator_broken", "v_insulator_chip_off"],
    "h_insulator":      ["h_insulator_normal", "h_insulator_broken", "h_insulator_chip_off"],
    "wire":             ["wire_normal", "cross_wire"],
    "crossarm_stright": ["straight_crossarm_normal", "straight_crossarm_band"],
    "top_crossarm":     ["top_crossarm_normal", "top_crossarm_band"],
    "om_crossarm":      ["om_crossarm_normal"],
    # vegetation, rust -> no condition family (presence/defect only)
}

# One place every consumer reads the per-subset class list from.
SUBSET_CLASSES = {
    "pole": POLE_CLASSES,
    "component_above_1000": COMPONENT_ABOVE_CLASSES,
    "component_below_1000": COMPONENT_BELOW_CLASSES,
    "component_classification": COMPONENT_CLASSIFICATION_CLASSES,
}
BASE_SUBSETS = list(SUBSET_CLASSES)

# Crop-aligned variants: train the component/condition detectors on CROPS at the same spatial
# scale they are run on at inference (above/below run on the pole crop; condition runs on the
# component crop), instead of on full ~4000x3000 frames. This closes the train/serve scale gap
# for thin wires and small insulators. Each <base>_crop subset shares its base's class list and
# balance/source/merge policy; keep BOTH and pick the val-mAP winner (full-frame vs crop ablation).
# CROP_ALIGN[base] = (mode, anchor_classes, pad_frac, min_visible_frac):
#   * "anchor": crop to each anchor box (the pole) + pad; keep in-subset boxes >= min_visible in the crop.
#   * "self":   crop to each in-subset box + pad (the component itself, with a little context).
# Padding around a COMPONENT when cropping it for the condition (classification) model. This is
# the SINGLE source of truth used BOTH when building component_classification_crop ("self" mode
# below) AND when the inference pipeline crops a detected component to feed the condition model,
# so train scale/context == serve scale/context. Subtle insulator defects (band/chip_off) need the
# surrounding context, and padding also tolerates imperfect detector boxes.
CONDITION_CROP_PAD = 0.25

CROP_ALIGN = {
    "component_above_1000": ("anchor", tuple(POLE_CLASSES), 0.05, 0.30),
    "component_below_1000": ("anchor", tuple(POLE_CLASSES), 0.05, 0.30),
    "component_classification": ("self", None, CONDITION_CROP_PAD, 0.50),
}
CROP_SUBSETS = [b + "_crop" for b in CROP_ALIGN]
for _b in CROP_ALIGN:
    SUBSET_CLASSES[_b + "_crop"] = SUBSET_CLASSES[_b]
SUBSETS = list(SUBSET_CLASSES)


def base_subset(subset: str) -> str:
    """The full-frame base a (possibly crop-aligned) subset derives its policy from."""
    return subset[:-len("_crop")] if subset.endswith("_crop") else subset

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

# For the per-object CONDITION subset, after merging the union of members' boxes, resolve
# the case where members gave the SAME physical object DIFFERENT condition labels: defect
# beats normal, and a defect-vs-defect disagreement is dropped as ambiguous. Only meaningful
# for component_classification (for the above/below detectors, overlapping distinct classes
# like a wire crossing a crossarm are legitimate, so it is left OFF there).
RESOLVE_CONDITION_CONFLICTS = {"component_classification": True}

# Split
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

# Grouping: new capture-sequence group when consecutive frames differ by > this (seconds)
GROUP_TIME_GAP_S = 60

# Balancing
BALANCE_CAP_ENABLED = True  # cap to lowest kept-class count per sub-dataset

# CLAHE defaults (per-image params still come from the profile)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
