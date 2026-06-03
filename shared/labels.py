"""Label source-of-truth = name-based VOC XML. Index-based YOLO .txt files are ignored."""

# All raw spellings seen in the data, mapped to a canonical kept class.
# Excluded classes (rust, om_crossarm, top_crossarm, vegetation) are intentionally
# NOT listed -> they normalize to None (ignored, never deleted from source).
_CANONICAL = {
    "pole": "pole",
    "wire": "wire",
    "h_insulator": "h_insulator",
    "v_insulator": "v_insulator",
    "crossarm_stright": "crossarm_stright",
    "crossarm_stright ": "crossarm_stright",
    "crossarmstright": "crossarm_stright",  # the mem5 stray 10th class
}


def normalize_class_name(raw: str):
    """Return canonical kept-class name, or None if the class is excluded/unknown."""
    if raw is None:
        return None
    key = raw.strip().lower()
    return _CANONICAL.get(key)


from dataclasses import dataclass
from pathlib import Path
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
