"""VOC XML is the label source of truth (index-based YOLO .txt is ignored).

Ported/adapted from the trainDronisight data_prep design, scoped to the two
thermal classes. Names are normalized so annotator typos / view-specific labels
collapse to the canonical class; anything unknown drops to None.
"""
import hashlib
from dataclasses import dataclass
from xml.etree import ElementTree as ET

# Every raw spelling seen in the 4-annotator thermal data -> canonical class.
_CANONICAL = {
    "transformer": "transformer",
    "front_transformer": "transformer",  # view-specific stray label
    "transform": "transformer",           # typo
    "wire": "wire",
}


def normalize_class_name(raw):
    if raw is None:
        return None
    return _CANONICAL.get(raw.strip().lower())


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
    """Parse a VOC XML, normalizing names and dropping excluded/degenerate boxes."""
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
            continue
        boxes.append(Box(name, xmin, ymin, xmax, ymax))
    return Annotation(width, height, boxes)


def to_yolo_line(box: Box, img_w: int, img_h: int, class_names) -> str:
    """One YOLO label line '<cls> <xc> <yc> <w> <h>' (normalized) for an in-class box."""
    cls = class_names.index(box.name)
    xc = ((box.xmin + box.xmax) / 2) / img_w
    yc = ((box.ymin + box.ymax) / 2) / img_h
    w = (box.xmax - box.xmin) / img_w
    h = (box.ymax - box.ymin) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def image_content_hash(path) -> str:
    """MD5 of raw image bytes: same content == same physical photo regardless of
    folder/filename. The key that collapses byte-identical duplicates (the _T/_T_1
    copies and cross-annotator copies) so they cannot leak across train/val/test."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
