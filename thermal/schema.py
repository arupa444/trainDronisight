from dataclasses import dataclass

# A box detected by YOLO. cls is "transformer" or "wire".
@dataclass
class Detection:
    cls: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    conf: float = 1.0

# One defect decision made by the CV layer.
@dataclass
class DefectFinding:
    component: str          # "transformer" or "wire"
    bbox: tuple[int, int, int, int]
    severity: str           # Normal / Watch / Investigate / Critical
    relative_delta: float   # how far above reference, in 0..1 intensity units
