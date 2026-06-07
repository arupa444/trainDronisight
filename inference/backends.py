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


def parse_torchvision_output(output, class_names, conf: float) -> list:
    """Convert a torchvision detection dict into Detections (label 1 -> class_names[0])."""
    dets = []
    boxes = output["boxes"].tolist()
    scores = output["scores"].tolist()
    labels = output["labels"].tolist()
    for box, score, label in zip(boxes, scores, labels):
        if score < conf:
            continue
        idx = int(label) - 1  # undo background offset
        if 0 <= idx < len(class_names):
            dets.append(Detection(class_names[idx], float(score),
                                  tuple(float(v) for v in box)))
    return dets


class TorchvisionDetector:
    """Wraps a trained Faster R-CNN state_dict behind the Detector interface."""

    def __init__(self, weights_path, class_names, conf=0.5, device=None):
        import torch
        from shared.device import select_device
        from train_faster_rcnn.model import build_fasterrcnn
        self.class_names = class_names
        self.conf = conf
        self.device = device or select_device()
        self.model = build_fasterrcnn(len(class_names))
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval().to(self.device)

    def predict(self, image) -> list:
        import torch
        from torchvision.transforms.functional import to_tensor
        from PIL import Image
        import numpy as np
        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])  # BGR->RGB
        with torch.no_grad():
            out = self.model([to_tensor(image).to(self.device)])[0]
        out = {k: v.cpu() for k, v in out.items()}
        return parse_torchvision_output(out, self.class_names, self.conf)


class RFDetrDetector:
    """Wraps an RF-DETR-L checkpoint behind the Detector interface.

    resolution must match training (multiple of 56). The pipeline feeds BGR arrays
    (cv2/load_oriented_bgr) but RF-DETR expects RGB, so we convert."""

    def __init__(self, weights_path, class_names, conf=0.5, resolution=728):
        from rfdetr import RFDETRLarge
        self.model = RFDETRLarge(pretrain_weights=weights_path, resolution=resolution)
        self.class_names = class_names
        self.conf = conf
        # predict() needs a shape divisible by block_size (32); resolution is only required
        # %56, so e.g. 1008 must be rounded to 1024 for inference.
        self.shape = round(resolution / 32) * 32

    def predict(self, image) -> list:
        import numpy as np
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            image = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB
        det = self.model.predict(image, threshold=self.conf, shape=self.shape)
        results = []
        for xyxy, conf, cls_id in zip(det.xyxy, det.confidence, det.class_id):
            i = int(cls_id)
            if 0 <= i < len(self.class_names):            # guard against id/label drift
                results.append(Detection(self.class_names[i], float(conf),
                                         tuple(float(v) for v in xyxy)))
        return results
