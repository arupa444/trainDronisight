"""Label source-of-truth = name-based VOC XML. Index-based YOLO .txt files are ignored."""

# All raw spellings seen in the data, mapped to a canonical kept class.
# Anything NOT listed normalizes to None (genuinely unknown classes only).
# The 4 rare classes (vegetation, top_crossarm, om_crossarm, rust) are now KEPT
# (trained as the `component_below_1000` detector) -- previously they were excluded.
_CANONICAL = {
    "pole": "pole",
    "wire": "wire",
    "h_insulator": "h_insulator",
    "v_insulator": "v_insulator",
    "crossarm_stright": "crossarm_stright",
    "crossarm_stright ": "crossarm_stright",
    "crossarmstright": "crossarm_stright",  # the mem5 stray 10th class
    "vegetation": "vegetation",
    "top_crossarm": "top_crossarm",
    "om_crossarm": "om_crossarm",
    "rust": "rust",
    # --- component-condition classes (6th june data) ---
    "straight_crossarm_normal": "straight_crossarm_normal",
    "straight_crossarm_band": "straight_crossarm_band",
    "v_insulator_normal": "v_insulator_normal",
    "wire_normal": "wire_normal",
    "h_insulator_normal": "h_insulator_normal",
    "cross_wire": "cross_wire",
    "h_insulator_broken": "h_insulator_broken",
    "v_insulator_band": "v_insulator_band",
    "v_insulator_broken": "v_insulator_broken",
    "top_crossarm_band": "top_crossarm_band",
    "h_insulator_chip_off": "h_insulator_chip_off",
    "om_crossarm_normal": "om_crossarm_normal",
    "v_insulator_chip_off": "v_insulator_chip_off",
    "top_crossarm_normal": "top_crossarm_normal",
    "top_corssarm_normal": "top_crossarm_normal",      # misspelling -> merge
    "v_insulator_puncture": "v_insulator_chip_off",    # merge punctures into chip_off
    "h_insulator_puncture": "h_insulator_chip_off",
    # 'w' (stray) and 'om_crossarm_band' (excluded) intentionally absent -> normalize to None
}


def normalize_class_name(raw: str):
    """Return canonical kept-class name, or None if the class is excluded/unknown."""
    if raw is None:
        return None
    key = raw.strip().lower()
    return _CANONICAL.get(key)


from dataclasses import dataclass
from xml.etree import ElementTree as ET


@dataclass
class Box:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Annotation:
    width: int
    height: int
    boxes: list  # list[Box]


def parse_voc(path) -> Annotation:
    """Parse a VOC XML, normalizing names and dropping excluded/invalid boxes."""
    root = ET.parse(str(path)).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        name = normalize_class_name(obj.findtext("name"))
        if name is None:
            continue
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin")))
        ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax")))
        ymax = int(float(bb.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue  # drop degenerate boxes
        boxes.append(Box(name, xmin, ymin, xmax, ymax))
    return Annotation(width, height, boxes)


def annotation_hash(boxes) -> str:
    """Content hash of an annotation: sha256 over the sorted (name, coords) tuples.
    Two annotations with the same set of boxes (any order) hash equal. Used to dedup
    re-annotated images that appear in more than one source folder."""
    import hashlib
    items = sorted((b.name, b.xmin, b.ymin, b.xmax, b.ymax) for b in boxes)
    return hashlib.sha256(repr(items).encode()).hexdigest()


def to_yolo_line(box: Box, img_w: int, img_h: int, class_names: list) -> str:
    """Format one box as a YOLO label line: '<cls> <xc> <yc> <w> <h>' (normalized)."""
    if box.name not in class_names:
        raise ValueError(f"{box.name!r} not in {class_names}")
    cls = class_names.index(box.name)
    xc = ((box.xmin + box.xmax) / 2) / img_w
    yc = ((box.ymin + box.ymax) / 2) / img_h
    w = (box.xmax - box.xmin) / img_w
    h = (box.ymax - box.ymin) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
