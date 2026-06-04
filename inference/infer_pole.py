"""Run Model 1 (pole) on one image. Usage:
    python -m inference.infer_pole --image x.jpg --weights pole.pt
Applies the same EXIF-orient + CLAHE preprocessing as training (use --no-clahe only
for a model trained on the 'orig' variant). Defaults to imgsz 640 to match pole training.
"""
import argparse
import json

from inference.backends import YoloDetector
from data_prep.preprocess import load_oriented_bgr, clahe_image


def detections_to_records(dets):
    return [{"class": d.class_name, "confidence": d.confidence,
             "box": [int(v) for v in d.box]} for d in dets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-clahe", action="store_true",
                    help="skip CLAHE (only for models trained on the 'orig' variant)")
    a = ap.parse_args()
    img = load_oriented_bgr(a.image)
    if not a.no_clahe:
        img = clahe_image(img)
    det = YoloDetector(a.weights, conf=a.conf, imgsz=a.imgsz)
    recs = detections_to_records(det.predict(img))
    print(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
