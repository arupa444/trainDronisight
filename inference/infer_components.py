"""Run Model 2 (components) on one image/crop. Usage:
    python -m inference.infer_components --image crop.jpg --weights components.pt
Applies the same EXIF-orient + CLAHE preprocessing as training (--no-clahe for an
'orig'-trained model). Defaults to imgsz 1280 to match component training (thin wires).
"""
import argparse
import json

from inference.backends import YoloDetector
from inference.infer_pole import detections_to_records
from data_prep.preprocess import load_oriented_bgr, clahe_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
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
