from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels of the image it was run on


def parse_yolo_result(result) -> list:
    """Convert one Ultralytics Result into Detection objects."""
    dets = []
    for b in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in list(b.xyxy[0]))
        cls = int(b.cls[0])
        dets.append(Detection(result.names[cls], float(b.conf[0]),
                              (x1, y1, x2, y2)))
    return dets


def filter_detections(dets, conf: float) -> list:
    return [d for d in dets if d.confidence >= conf]


class Detector(Protocol):
    def predict(self, image) -> list: ...


class YoloDetector:
    """Wraps an Ultralytics model behind the Detector interface."""

    def __init__(self, weights, conf=0.25, imgsz=1280):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, image) -> list:
        res = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        return parse_yolo_result(res)
