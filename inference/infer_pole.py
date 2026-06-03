"""Run Model 1 (pole) on one image. Usage:
    python -m inference.infer_pole --image x.jpg --weights pole.pt
"""
import argparse
import json

import cv2

from inference.backends import YoloDetector


def detections_to_records(dets):
    return [{"class": d.class_name, "confidence": d.confidence,
             "box": [int(v) for v in d.box]} for d in dets]


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
