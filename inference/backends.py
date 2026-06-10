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


class FilteredDetector:
    """Wrap a Detector and keep only detections whose class_name is in `keep`. Lets us reuse a
    multi-class detector for a single class (e.g. the below_1000 detector's `vegetation`) without
    its other classes leaking into the pipeline."""

    def __init__(self, inner, keep):
        self.inner = inner
        self.keep = set(keep)

    def predict(self, image) -> list:
        return [d for d in self.inner.predict(image) if d.class_name in self.keep]


class EnsembleDetector:
    """Run several detectors on the same image and UNION their detections. Used to combine a condition
    specialist with the old unified classifier so a defect fires if EITHER model sees it (recall up).
    Same-class duplicates from the two models are collapsed downstream (resolve_condition_overlaps)."""

    def __init__(self, detectors):
        self.detectors = list(detectors)

    def predict(self, image) -> list:
        out = []
        for d in self.detectors:
            out.extend(d.predict(image))
        return out


class YoloDetector:
    """Wraps an Ultralytics model behind the Detector interface.

    device follows the CUDA -> MPS -> CPU priority (shared.device.select_device). Ultralytics
    auto-uses CUDA but defaults to CPU for predict otherwise, so we pass the resolved device
    explicitly — this is what makes the model actually run on Apple-Silicon MPS."""

    def __init__(self, weights, conf=0.25, imgsz=1280, device=None):
        from ultralytics import YOLO
        from shared.device import select_device
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or select_device()

    def predict(self, image) -> list:
        res = self.model.predict(image, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]
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
    """Wraps a trained Faster R-CNN state_dict behind the Detector interface.

    min_size/max_size MUST match training (train_faster_rcnn.train defaults: 2000/3000), or
    the GeneralizedRCNNTransform serves objects at a different scale than trained and small
    objects (thin wires/insulators) collapse. Defaults here mirror the trainer; override if
    you trained with a different --min-size."""

    def __init__(self, weights_path, class_names, conf=0.5, device=None,
                 min_size=2000, max_size=None):
        import torch
        from shared.device import select_device
        from train_faster_rcnn.model import build_fasterrcnn
        self.class_names = class_names
        self.conf = conf
        self.device = device or select_device()
        max_size = max_size or round(min_size * 1.5)   # mirrors train.py's max_size rule
        self.model = build_fasterrcnn(len(class_names), min_size=min_size, max_size=max_size)
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


def rfdetr_block_size(default=32):
    """RF-DETR's required resolution divisor = patch_size * num_windows for the L variant.
    Read from the installed library (version-robust): older RF-DETR-L used 14*4=56, the
    current build uses 16*2=32. Both training resolution and predict shape must be a multiple."""
    try:
        from rfdetr.config import RFDETRLargeConfig
        c = RFDETRLargeConfig()
        return int(c.patch_size) * int(c.num_windows)
    except Exception:
        return default


class RFDetrDetector:
    """Wraps an RF-DETR-L checkpoint behind the Detector interface.

    resolution must be a multiple of the model's block_size (patch_size*num_windows; 32 on
    the installed build). The pipeline feeds BGR arrays (cv2/load_oriented_bgr) but RF-DETR
    expects RGB, so we convert."""

    def __init__(self, weights_path, class_names, conf=0.5, resolution=672):
        from rfdetr import RFDETRLarge
        self.model = RFDETRLarge(pretrain_weights=weights_path, resolution=resolution)
        self.class_names = class_names
        self.conf = conf
        # predict() validates shape % block_size == 0 on BOTH dims and wants a (h, w) tuple.
        block = rfdetr_block_size()
        self.shape = max(block, round(resolution / block) * block)

    def predict(self, image) -> list:
        import numpy as np
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            image = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB
        det = self.model.predict(image, threshold=self.conf, shape=(self.shape, self.shape))
        results = []
        for xyxy, conf, cls_id in zip(det.xyxy, det.confidence, det.class_id):
            i = int(cls_id)
            if 0 <= i < len(self.class_names):            # guard against id/label drift
                results.append(Detection(self.class_names[i], float(conf),
                                         tuple(float(v) for v in xyxy)))
        return results
