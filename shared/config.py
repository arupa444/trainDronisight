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
# Stage-2 component detectors are segregated by visual NATURE/SCALE (each run on the pole crop):
#   comp_wire       : thin lines (its own thin-object scale regime)
#   comp_insulator  : h_insulator + v_insulator (small mounted; discriminate orientation)
#   comp_crossarm   : crossarm_stright + top_crossarm + om_crossarm (elongated bars; one softmax so
#                     they stop being mis-typed -- the om_crossarm<->straight bug)
#   comp_vegetation : large diffuse foliage
#   comp_rust       : corrosion texture (tiny; data-limited)
COMP_WIRE_CLASSES       = ["wire"]
COMP_INSULATOR_CLASSES  = ["h_insulator", "v_insulator"]
COMP_CROSSARM_CLASSES   = ["crossarm_stright", "top_crossarm", "om_crossarm"]
COMP_VEGETATION_CLASSES = ["vegetation"]
COMP_RUST_CLASSES       = ["rust"]
COMP_SUBSETS = ["comp_wire", "comp_insulator", "comp_crossarm", "comp_vegetation", "comp_rust"]
COMPONENT_CLASSES = (COMP_WIRE_CLASSES + COMP_INSULATOR_CLASSES + COMP_CROSSARM_CLASSES
                     + COMP_VEGETATION_CLASSES + COMP_RUST_CLASSES)   # the 8 types (union, for reference)

# A component slot may be served by a DIFFERENT detector, filtered to only the listed classes.
# The solo comp_vegetation single-class model over-fires (no competing class to lose detections to,
# so it labels poles/crossarms as vegetation). The older component_below_1000 detector learned
# vegetation AGAINST top_crossarm/om_crossarm/rust, so it discriminates better -> use it but KEEP
# ONLY vegetation (the other 3 classes are owned by comp_crossarm / comp_rust and must not leak in).
# Falls back to the slot's own weights if the override weights aren't found.
COMPONENT_WEIGHTS_OVERRIDE = {
    "comp_vegetation": {"weights_subset": "component_below_1000_crop", "keep": ["vegetation"]},
}

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
COND_SUBSETS = ["cond_v_insulator", "cond_h_insulator", "cond_straight_crossarm",
                "cond_top_crossarm", "cond_om_crossarm", "cond_wire"]
SUBSET_CLASSES = {
    "pole": POLE_CLASSES,
    "comp_wire": COMP_WIRE_CLASSES,
    "comp_insulator": COMP_INSULATOR_CLASSES,
    "comp_crossarm": COMP_CROSSARM_CLASSES,
    "comp_vegetation": COMP_VEGETATION_CLASSES,
    "comp_rust": COMP_RUST_CLASSES,
    "cond_v_insulator": COND_V_INSULATOR_CLASSES,
    "cond_h_insulator": COND_H_INSULATOR_CLASSES,
    "cond_straight_crossarm": COND_STRAIGHT_CROSSARM_CLASSES,
    "cond_top_crossarm": COND_TOP_CROSSARM_CLASSES,
    "cond_om_crossarm": COND_OM_CROSSARM_CLASSES,
    "cond_wire": COND_WIRE_CLASSES,
}
SUBSETS = list(SUBSET_CLASSES)

# component -> the condition classes valid for it (derived from the per-family models); used by the
# pipeline to keep only in-family condition detections. vegetation/rust absent -> no condition.
COMPONENT_TO_CONDITIONS = {c: SUBSET_CLASSES[m] for c, m in COMPONENT_TO_CONDITION_MODEL.items()}

# Crop padding (single source of truth so BUILD scale == inference scale):
POLE_CROP_PAD = 0.05        # `component` detector: crop to the pole box + this pad (== inference --pole-pad)
# BUILD pad for cond_* datasets. Annotators boxed each component down to the hinge, so 0.25 around that
# GT box already includes the band. CROP_ALIGN uses this -> do NOT change without rebuilding cond_*.
CONDITION_CROP_PAD = 0.25
# INFERENCE pad for the condition crop. It is LARGER than the build pad on purpose: the component
# DETECTOR boxes mostly the insulator HEAD (tighter than the hinge-to-hinge GT box), so detector_box +
# 0.25 would clip the band. detector_box + 0.40 (~0.25 trained + 0.15 extra reach) restores the
# band/hinge context and better matches the training distribution. No rebuild needed; tune via
# --condition-pad.
CONDITION_INFER_PAD = 0.40
# Saved component crop / thumbnail padding (display + the crop_path in result.csv) so a thin edge band
# isn't clipped from view. Presentation ONLY — the condition MODEL is fed the CONDITION_INFER_PAD crop.
COMPONENT_CROP_PAD = 0.15

# Inference de-duplication thresholds:
POLE_NMS_IOU = 0.5               # drop duplicate pole boxes on the same pole (keep highest confidence)
COMPONENT_NMS_IOU = 0.55         # DIFFERENT-class overlap (wire crossing a crossarm) kept unless this high
COMPONENT_SAME_CLASS_IOU = 0.45  # SAME-class component duplicates removed more aggressively
CONDITION_OVERLAP_IOS = 0.5      # condition normal-vs-damage conflict, by intersection-over-smaller-box

# How each subset is crop-aligned at BUILD time. A subset is crop-trained IFF it appears here.
#   "anchor": crop to each anchor (pole) box + pad; keep in-subset boxes >= min_visible.
#   "self":   crop to each in-subset box + pad (the component itself).
# `pole` is absent -> trained on the full frame.
CROP_ALIGN = {
    **{s: ("anchor", tuple(POLE_CLASSES), POLE_CROP_PAD, 0.30) for s in COMP_SUBSETS},
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
    **{s: SOURCE_DIRS for s in COMP_SUBSETS},
    **{s: CONDITION_SOURCE_DIRS for s in COND_SUBSETS},
}

# Balance policy (TRAIN only; val/test raw). Earlier fixed targets (400/1000) CAPPED the majority
# class and discarded most real data (e.g. straight_crossarm 1212+1158 -> 400+400 lost 1570 for no
# benefit). Instead: KEEP ALL real data and OVERSAMPLE the rarer classes UP toward the MAX class
# (data_prep.oversample, target=None) for every MULTI-class subset. MAX_OVERSAMPLE_FACTOR bounds the
# TOTAL augmented copies at factor x (#train images) -- a runaway guard, not a per-class cap. (A
# very rare class like wire_normal ~62 still gets lifted toward its family max; the real cure for
# those is more labels, not more duplicates.) Single-class detectors (wire/vegetation/rust/pole): none.
BALANCE_TARGET = {}                 # no capping of any class
MAX_OVERSAMPLE_FACTOR = 6

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
