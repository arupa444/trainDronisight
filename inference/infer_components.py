"""Run Model 2 (components) on one image/crop. Usage:
    python -m inference.infer_components --image crop.jpg --weights components.pt
"""
import argparse
import json

import cv2

from inference.backends import YoloDetector
from inference.infer_pole import detections_to_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    det = YoloDetector(a.weights, conf=a.conf)
    recs = detections_to_records(det.predict(cv2.imread(a.image)))
    print(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
