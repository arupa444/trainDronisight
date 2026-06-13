import numpy as np
from thermal.schema import Detection

# Detector confidence floor. Lowered below Ultralytics' 0.25 default because transformers
# that are distant/occluded/cluttered are detected only weakly (~0.12-0.15 conf) and the
# 0.25 cutoff dropped them entirely (missing the whole unit -> missing its faults). 0.12
# trades some precision (occasional spurious box) for recall. Tunable per deployment.
_DEFAULT_CONF = 0.12


class YoloDetector:
    """Generic single-model YOLO wrapper -> list[Detection]. Used for BOTH cascade
    stages: one instance loads the transformer weights, another the wire weights."""

    def __init__(self, weights_path: str):
        from ultralytics import YOLO  # lazy import keeps torch out of the test path
        self.model = YOLO(weights_path)
        self.names = self.model.names

    def detect(self, img_bgr: np.ndarray,
               conf: float = _DEFAULT_CONF) -> list[Detection]:
        # img_bgr: BGR uint8 (cv2/Ultralytics convention — same order training used)
        result = self.model(img_bgr, conf=conf)[0]
        dets = []
        for box in result.boxes:
            cls_name = self.names[int(box.cls[0])]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            dets.append(Detection(cls_name, (x1, y1, x2, y2), float(box.conf[0])))
        return dets
